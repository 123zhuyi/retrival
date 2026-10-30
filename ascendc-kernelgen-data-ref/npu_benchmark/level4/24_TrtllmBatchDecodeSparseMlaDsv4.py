import json
import os

import torch
import torch.nn as nn

DSV4_HEAD_DIM = 512
DSV4_SWA_TOPK = 128  # sparse_indices 前 128 列固定为 SWA 段


# ---------------------------------------------------------------------------
# TrtllmBatchDecodeSparseMlaDsv4 —— TRTLLM-GEN DeepSeek-V4 稀疏 MLA decode（融合版）
#
# 借鉴 ops-ref `sparse_flash_mla` AscendC kernel 的融合算法（SWA + CSA 双池）：
#   O = softmax(Q @ K~^T · bmm1_scale) @ K~ · bmm2_scale，K~ 由两个分页 KV 池
#   （swa_kv_cache + compressed_kv_cache）按 sparse_indices 稀疏 gather 拼接。
# 原版逐 query token 循环，融合版一次向量化全部请求：
#   1. sparse_indices [sum_q, 128+topk] 一次 gather：SWA 段索引 swa 池、压缩段
#      索引 comp 池，cat 成 [sum_q, Ktot, 512]；无效项由 (index>=0 & <pool 大小)
#      + sparse_topk_lens 前缀长度共同掩码（sparse_flash_mla ori_topk_length 语义）。
#   2. 无效 KV 槽置 0 后一次 batched matmul 得全部 logits，无效位置 -inf。
#   3. 稳定 softmax（含全无效行守卫）@ kv，sinks 修正，输出恒 BF16。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, swa_kv_cache, compressed_kv_cache, sparse_indices,
                sparse_topk_lens, seq_lens, bmm1_scale=1.0, bmm2_scale=1.0,
                sinks=None, cum_seq_lens_q=None):
        D = DSV4_HEAD_DIM
        device = query.device
        # 任意分页布局 (HND/NHD, 3D/4D) 扁平化为 [num_pages * page_size, 512]
        swa_flat = swa_kv_cache.reshape(-1, D).float()
        comp_flat = compressed_kv_cache.reshape(-1, D).float()

        if query.ndim == 4:
            B, q_len, H, _ = query.shape
            sum_q = B * q_len
            out_shape = (B, q_len, H, D)
        else:
            sum_q, H, _ = query.shape
            out_shape = (sum_q, H, D)

        q_flat = query.reshape(sum_q, H, D).float()

        # ---- 1. sparse_indices 一次向量化 gather（SWA 段 + 压缩段）----
        swa_idx = sparse_indices[:, :DSV4_SWA_TOPK].long()           # [sum_q, 128]
        comp_idx = sparse_indices[:, DSV4_SWA_TOPK:].long()          # [sum_q, comp_topk]
        swa_valid = (swa_idx >= 0) & (swa_idx < swa_flat.shape[0])
        swa_g = swa_flat[swa_idx.clamp(0, swa_flat.shape[0] - 1)]    # [sum_q, 128, 512]
        if comp_idx.numel() > 0:
            comp_valid_len = (sparse_topk_lens - DSV4_SWA_TOPK).clamp(min=0)  # [sum_q]
            comp_valid = (comp_idx >= 0) & (comp_idx < comp_flat.shape[0])
            comp_valid &= (torch.arange(comp_idx.shape[1], device=device)
                           < comp_valid_len.unsqueeze(-1))
            comp_g = comp_flat[comp_idx.clamp(0, comp_flat.shape[0] - 1)]
            kv = torch.cat([swa_g, comp_g], dim=1)                   # [sum_q, Ktot, 512]
            valid = torch.cat([swa_valid, comp_valid], dim=1)        # [sum_q, Ktot]
        else:
            kv = swa_g                                               # [sum_q, 128, 512]
            valid = swa_valid                                        # [sum_q, 128]

        # ---- 2. 无效槽置 0 + 一次 batched matmul + -inf 掩码 ----
        kv_masked = torch.where(valid.unsqueeze(-1), kv, torch.zeros_like(kv))
        logits = (q_flat @ kv_masked.transpose(-1, -2)) * bmm1_scale  # [sum_q, H, Ktot]
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))

        # ---- 3. 稳定 softmax（含全无效行守卫）+ @kv + sinks ----
        m = logits.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
        e = torch.exp(logits - m)
        denom = e.sum(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        lse = m.squeeze(-1) + torch.log(denom.squeeze(-1))            # [sum_q, H]
        out = (e / denom) @ kv_masked

        if sinks is not None:
            # 等价于 softmax 分母附加 exp(sinks): o *= 1/(1+exp(sinks-lse))
            sink_scale = 1.0 / (1.0 + torch.exp(sinks.float() - lse))
            sink_scale = torch.where(torch.isfinite(sink_scale), sink_scale,
                                     torch.zeros_like(sink_scale))
            out = out * sink_scale.unsqueeze(-1)

        return (out * bmm2_scale).reshape(out_shape).to(torch.bfloat16)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__),
                             "24_TrtllmBatchDecodeSparseMlaDsv4.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # 输入分布族与原版一致（50% 均匀 + 50% 正态），幅值缩至数值稳定区间（~1.0）
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0)
        else:
            mu = float(torch.empty(1).uniform_(-0.2, 0.2).item())
            sigma = float(torch.empty(1).uniform_(0.1, 0.3).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }

    def make_block_table(npages, total_pages):
        perm = torch.randperm(total_pages)
        max_pages = max(npages)
        bt = torch.zeros(len(npages), max_pages, dtype=torch.int64)
        off = 0
        for b, n in enumerate(npages):
            bt[b, :n] = perm[off:off + n]
            off += n
        return bt

    def abs_to_flat(abs_idx, bt_row, ps):
        safe = abs_idx.clamp_min(0)
        flat = bt_row[safe // ps] * ps + (safe % ps)
        flat[abs_idx < 0] = -1
        return flat

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        kv_lens = qo_lens = None
        swa_ps = comp_ps = comp_ratio = comp_topk = None
        bmm1_scale, bmm2_scale = 1.0, 1.0
        has_sinks = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "qo_lens":
                qo_lens = None if inp["value"] is None else inp["value"]
            elif name == "swa_page_size":
                swa_ps = inp["value"]
            elif name == "compressed_page_size":
                comp_ps = inp["value"]
            elif name == "compressed_ratio":
                comp_ratio = inp["value"]
            elif name == "compressed_topk":
                comp_topk = inp["value"]
            elif name == "bmm1_scale":
                bmm1_scale = inp["value"]
            elif name == "bmm2_scale":
                bmm2_scale = inp["value"]
            elif name == "has_sinks":
                has_sinks = inp["value"]

        B = len(kv_lens)
        D = DSV4_HEAD_DIM
        H = shapes["query"][-2]
        capacity = DSV4_SWA_TOPK + comp_topk
        assert capacity % 4 == 0

        if qo_lens is None:
            q_len = shapes["query"][1]
            qo_lens = [q_len] * B
            cum_seq_lens_q = None
            sum_q = B * q_len
        else:
            cum_seq_lens_q = torch.tensor(
                [0] + [sum(qo_lens[:i + 1]) for i in range(B)], dtype=torch.int32)
            sum_q = sum(qo_lens)
        assert all(l >= q for l, q in zip(kv_lens, qo_lens))

        query = random_tensor(shapes["query"], dtypes["query"])
        swa_kv_cache = random_tensor(shapes["swa_kv_cache"], dtypes["swa_kv_cache"])
        compressed_kv_cache = random_tensor(shapes["compressed_kv_cache"],
                                            dtypes["compressed_kv_cache"])
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32)

        swa_npages = [(l + swa_ps - 1) // swa_ps for l in kv_lens]
        swa_total = sum(swa_npages) + 3
        assert shapes["swa_kv_cache"][0] == swa_total
        swa_bt = make_block_table(swa_npages, swa_total)

        c_lens = [(l + comp_ratio - 1) // comp_ratio for l in kv_lens] \
            if comp_topk > 0 else [0] * B
        comp_npages = [max((l + comp_ps - 1) // comp_ps, 0) for l in c_lens]
        if comp_topk > 0:
            comp_total = sum(comp_npages) + 3
            assert shapes["compressed_kv_cache"][0] == comp_total
            comp_bt = make_block_table(comp_npages, comp_total)
        else:
            comp_bt = None

        sparse_indices = torch.full((sum_q, capacity), -1, dtype=torch.int32)
        sparse_topk_lens = torch.full((sum_q,), DSV4_SWA_TOPK, dtype=torch.int32)
        for b in range(B):
            kv_len, q_len_b = kv_lens[b], qo_lens[b]
            for t in range(q_len_b):
                j = (b * qo_lens[0] + t) if cum_seq_lens_q is None \
                    else (int(cum_seq_lens_q[b].item()) + t)
                token_idx = kv_len - q_len_b + t
                n_swa = min(DSV4_SWA_TOPK, token_idx + 1)
                abs_swa = torch.arange(token_idx - n_swa + 1, token_idx + 1,
                                       dtype=torch.int64)
                flat_swa = abs_to_flat(abs_swa, swa_bt[b], swa_ps)
                sparse_indices[j, :DSV4_SWA_TOPK] = torch.cat([
                    flat_swa.int(),
                    torch.full((DSV4_SWA_TOPK - n_swa,), -1, dtype=torch.int32)])
                if comp_topk > 0:
                    base = min(comp_topk, c_lens[b])
                    n_comp = max(base - (b + t) % 4, 0)
                    if n_comp > 0:
                        abs_comp = torch.randperm(c_lens[b])[:n_comp]
                        flat_comp = abs_to_flat(abs_comp, comp_bt[b], comp_ps)
                        sparse_indices[j, DSV4_SWA_TOPK:
                                       DSV4_SWA_TOPK + n_comp] = flat_comp.int()
                    sparse_topk_lens[j] = DSV4_SWA_TOPK + n_comp

        sinks = random_tensor([H], torch.float32) if has_sinks else None

        input_groups.append([query, swa_kv_cache, compressed_kv_cache,
                             sparse_indices, sparse_topk_lens, seq_lens,
                             bmm1_scale, bmm2_scale, sinks, cum_seq_lens_q])
    return input_groups


def get_init_inputs():
    return []
