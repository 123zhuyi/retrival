import json
import os

import torch
import torch.nn as nn

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# TrtllmBatchDecodeWithKvCacheMla —— TRTLLM-GEN 批量 MLA decode（页式，矩阵吸收）融合版
#
# 借鉴 ops-ref `sparse_flash_mla` + `mla_preprocess` AscendC kernel 的融合算法：
#   O = softmax(Q @ K~^T · bmm1_scale) @ ckv · bmm2_scale
#   K~ = concat([ckv, kpe])（query/kv_cache 均为 [nope|pe] 拼接）。
# 融合要点：
#   1. 双分支打分合并为一次 matmul：query [q_nope|q_pe]、kv [ckv|kpe] 沿 head_dim
#      拼接后一次 Q@K~^T（sparse_flash_mla 的 Q@K~^T 语义），省掉两个独立 einsum。
#   2. 稠密路径：block_tables 前 ceil(kv_len/page_size) 页一次 gather + padding。
#   3. 稀疏路径（sparse_mla_top_k>0）：block_tables [TQ, topk] 稀疏页索引，
#      无效项（<0 / 越界）一次 tok_valid 掩码屏蔽，clamp 后一次 gather。
#   4. req_id/t_id 把 ragged query 拉直为 [TQ, H, 576]；base-2 稳定 softmax + LSE。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, kv_cache, block_tables, seq_lens,
                kv_lora_rank, qk_rope_head_dim,
                bmm1_scale=1.0, bmm2_scale=1.0, sparse_mla_top_k=0,
                cum_seq_lens_q=None, return_lse=False, return_lse_base_on_e=False):
        head_dim_qk = kv_lora_rank + qk_rope_head_dim
        cache = kv_cache.squeeze(1) if kv_cache.dim() == 4 else kv_cache
        page_size = cache.shape[1]
        device = query.device
        R = kv_lora_rank

        # 统一将 query 拉直为 [TQ, H, 576]; req_id 记录每个 q token 所属请求
        if cum_seq_lens_q is None:
            B, q_len, H, _ = query.shape
            q_flat = query.reshape(B * q_len, H, head_dim_qk).float()
            req_id = torch.arange(B, device=device).repeat_interleave(q_len)
            t_id = torch.arange(q_len, device=device).repeat(B)
        else:
            B = cum_seq_lens_q.numel() - 1
            H = query.shape[1]
            q_flat = query.float()
            cu = cum_seq_lens_q.long()
            req_id = torch.repeat_interleave(
                torch.arange(B, device=device), cu[1:] - cu[:-1])
            t_id = torch.arange(q_flat.shape[0], device=device) - cu[:-1].long()[req_id]
        TQ = q_flat.shape[0]

        if sparse_mla_top_k > 0:
            # 稀疏路径：每个 query token 独立的稀疏页索引，无效项（<0/越界）屏蔽
            pages = block_tables.long()[req_id, t_id]                 # [TQ, topk]
            tok_valid = (pages >= 0) & (pages < cache.shape[0])       # [TQ, topk]
            pg = pages.clamp(0, cache.shape[0] - 1)
            flat = cache[pg].float().reshape(TQ, -1, head_dim_qk)     # [TQ, topk*page, 576]
            kv_mask = tok_valid.repeat_interleave(page_size, dim=1)   # [TQ, S]
        else:
            # 稠密路径：页表前 n_pages 项，gather + padding 成 [B, L, 576]
            kv_lens = seq_lens.long()
            npages = (kv_lens + page_size - 1) // page_size
            P = int(npages.max().item())
            L = int(kv_lens.max().item())
            pages = block_tables[:, :P].long()                        # [B, P]
            flat = cache[pages].float().reshape(B, P * page_size,
                                                head_dim_qk)[:, :L]   # [B, L, 576]
            kv_mask = (torch.arange(L, device=device)[None, :]
                       < kv_lens[:, None])                            # [B, L]
            kv_mask = kv_mask[req_id]                                 # [TQ, L]
            flat = flat[req_id]                                       # [TQ, L, 576]

        # ---- 双分支打分合并：K~ = concat([ckv, kpe]) 一次 Q@K~^T（sparse_flash_mla）----
        Q = q_flat                                                 # [TQ, H, 576] (nope|pe 拼接)
        logits = torch.einsum('thr,tsr->ths', Q, flat) * bmm1_scale
        logits = logits.masked_fill(~kv_mask[:, None, :], float("-inf"))

        # base-2 softmax; 无有效页的行输出 0 / lse 0
        s2 = logits * LOG2E
        m = s2.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
        e = torch.exp2(s2 - m)
        denom = e.sum(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        Kn = flat[..., :R]                                        # ckv：同时作为 K 和 V
        out = (e / denom) @ Kn * bmm2_scale                       # [TQ, H, R]
        lse2 = m.squeeze(-1) + torch.log2(denom.squeeze(-1))      # [TQ, H]

        o = out.to(query.dtype)
        lse = lse2 / LOG2E if return_lse_base_on_e else lse2
        if cum_seq_lens_q is None:
            o = o.reshape(B, q_len, H, R)
            lse = lse.reshape(B, q_len, H)
        if return_lse:
            return o, lse
        return o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "23_TrtllmBatchDecodeWithKvCacheMla.json")
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
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        kv_lens = None
        page_size = None
        kv_lora_rank = None
        qk_rope_head_dim = None
        bmm1_scale = 1.0
        bmm2_scale = 1.0
        sparse_mla_top_k = 0
        qo_lens = None
        return_lse = False
        return_lse_base_on_e = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "page_size":
                page_size = inp["value"]
            elif name == "kv_lora_rank":
                kv_lora_rank = inp["value"]
            elif name == "qk_rope_head_dim":
                qk_rope_head_dim = inp["value"]
            elif name == "bmm1_scale":
                bmm1_scale = inp["value"]
            elif name == "bmm2_scale":
                bmm2_scale = inp["value"]
            elif name == "sparse_mla_top_k":
                sparse_mla_top_k = inp["value"]
            elif name == "qo_lens":
                qo_lens = None if inp["value"] is None else inp["value"]
            elif name == "return_lse":
                return_lse = inp["value"]
            elif name == "return_lse_base_on_e":
                return_lse_base_on_e = inp["value"]

        query = random_tensor(shapes["query"], dtypes["query"])
        kv_cache = random_tensor(shapes["kv_cache"], dtypes["kv_cache"])
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
        cum_seq_lens_q = None
        if qo_lens is not None:
            cum_seq_lens_q = torch.tensor(
                [0] + [sum(qo_lens[:i + 1]) for i in range(len(qo_lens))], dtype=torch.int32)

        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        total_pages = sum(npages) + 3
        assert shapes["kv_cache"][0] == total_pages and shapes["kv_cache"][-2] == page_size

        if sparse_mla_top_k > 0:
            B, q_len = shapes["block_tables"][0], shapes["block_tables"][1]
            block_tables = torch.zeros(B, q_len, sparse_mla_top_k, dtype=torch.int32)
            for b in range(B):
                for t in range(q_len):
                    n_valid = sparse_mla_top_k - (b + t) % 2
                    pages = torch.randperm(total_pages)[:n_valid]
                    block_tables[b, t, :n_valid] = pages.int()
                    block_tables[b, t, n_valid:] = -1
        else:
            max_pages = max(npages)
            perm = torch.randperm(total_pages)
            block_tables = torch.zeros(len(kv_lens), max_pages, dtype=torch.int32)
            off = 0
            for b, n in enumerate(npages):
                block_tables[b, :n] = perm[off:off + n].int()
                off += n
            assert shapes["block_tables"] == [len(kv_lens), max_pages]

        input_groups.append([query, kv_cache, block_tables, seq_lens,
                             kv_lora_rank, qk_rope_head_dim,
                             bmm1_scale, bmm2_scale, sparse_mla_top_k,
                             cum_seq_lens_q, return_lse, return_lse_base_on_e])
    return input_groups


def get_init_inputs():
    return []
