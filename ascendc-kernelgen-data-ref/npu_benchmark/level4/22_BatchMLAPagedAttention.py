import json
import os

import torch
import torch.nn as nn

LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# BatchMLAPagedAttention —— 批量 MLA 分页注意力（MTP / decode，矩阵吸收）融合版
#
# 借鉴 ops-ref `sparse_flash_mla` + `mla_preprocess` AscendC kernel 的融合算法：
#   O = softmax(Q @ K~^T · softmax_scale) @ V~，K~ = V~ = ckv（Matrix Absorption）。
# 原版逐 batch / 逐 query token 嵌套循环，融合版一次向量化全部请求：
#   1. page gather 一次向量化：kv_indptr/kv_indices 页表 -> 每请求前 npages 页，
#      `kv_indices[flat]` 一次高级索引取出全部 KV 页，按 (P, page_size, Hc)
#      展平为 slot 维（sparse_flash_mla PA_BBND 语义）。
#   2. req_id / local_i 映射：把每个 query token 归属到 batch 与块内序号，
#      `ckv[req_id]` 广播到 [total_q, L, D]，一次 batched matmul 得到全部 scores。
#   3. 双分支打分 + scale 融合进 Q（d_scale_q）。
#   4. causal + padding mask 一次向量化生成（causal 截止列 = kv_len - qo_len + i）。
#   5. base-2 稳定 softmax（exp2 / log2，与 kernel 内部 log 底数一致）+ LSE。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache,
                qo_indptr, kv_indptr, kv_indices, kv_len_arr,
                causal, sm_scale, return_lse=False, return_lse_base_on_e=False):
        total_q, H, Dc = q_nope.shape
        Dp = q_pe.shape[-1]
        page_size = ckv_cache.shape[1]
        device = q_nope.device

        # ---- 1. page gather 一次向量化：每请求前 npages 页，展平为 slot 维 ----
        kv_lens = kv_len_arr.long()
        B = kv_lens.numel()
        qo_lens = qo_indptr[1:] - qo_indptr[:-1]                      # [B]
        npages = (kv_lens + page_size - 1) // page_size
        P = int(npages.max().item())
        L = int(kv_lens.max().item())
        Hc = ckv_cache.shape[2]
        j = torch.arange(P, device=device)
        flat = (kv_indptr[:-1, None] + j[None, :]).clamp(max=kv_indices.numel() - 1)
        pages = kv_indices.long()[flat]                               # [B, P]
        ckv = ckv_cache[pages].float() \
            .reshape(B, P * page_size * Hc, Dc)[:, :L]                # [B, L, Dc]
        kpe = kpe_cache[pages].float() \
            .reshape(B, P * page_size * Hc, Dp)[:, :L]                # [B, L, Dp]

        # ---- 2. req_id / local_i 映射 + 广播到全部 query token ----
        req_id = torch.repeat_interleave(torch.arange(B, device=device), qo_lens)
        local_i = torch.arange(total_q, device=device) - qo_indptr.long()[req_id]
        ckv_t = ckv[req_id]                                           # [total_q, L, Dc]
        kpe_t = kpe[req_id]                                           # [total_q, L, Dp]

        # ---- 3. 双分支打分 + scale 融合进 Q（d_scale_q）----
        s = (q_nope.float() * sm_scale) @ ckv_t.transpose(1, 2) \
            + (q_pe.float() * sm_scale) @ kpe_t.transpose(1, 2)       # [total_q, H, L]

        # ---- 4. causal + padding mask 一次生成 ----
        col = torch.arange(L, device=device)
        mask = col[None, :] >= kv_lens[req_id, None]                  # padding
        if causal:
            cutoff = (kv_lens - qo_lens)[req_id] + local_i            # 每 token 截止列
            mask = mask | (col[None, :] > cutoff[:, None])
        s = s.masked_fill(mask[:, None, :], float("-inf"))

        # ---- 5. base-2 稳定 softmax + @ckv + LSE ----
        s2 = s * LOG2E
        m = s2.max(dim=-1, keepdim=True).values
        e = torch.exp2(s2 - m)
        denom = e.sum(dim=-1, keepdim=True)
        o = (e / denom) @ ckv_t                                       # [total_q, H, Dc]
        lse2 = m.squeeze(-1) + torch.log2(denom.squeeze(-1))          # [total_q, H]

        o = o.to(q_nope.dtype)
        lse = lse2 / LOG2E if return_lse_base_on_e else lse2
        if return_lse:
            return o, lse
        return o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "22_BatchMLAPagedAttention.json")
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
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        qo_lens = None
        kv_lens = None
        page_size = None
        causal = None
        sm_scale = None
        return_lse = False
        return_lse_base_on_e = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "qo_lens":
                qo_lens = inp["value"]
            elif name == "kv_lens":
                kv_lens = inp["value"]
            elif name == "page_size":
                page_size = inp["value"]
            elif name == "causal":
                causal = inp["value"]
            elif name == "sm_scale":
                sm_scale = inp["value"]
            elif name == "return_lse":
                return_lse = inp["value"]
            elif name == "return_lse_base_on_e":
                return_lse_base_on_e = inp["value"]

        q_nope = random_tensor(shapes["q_nope"], dtypes["q_nope"])
        q_pe = random_tensor(shapes["q_pe"], dtypes["q_pe"])
        qo_indptr = torch.tensor([0] + [sum(qo_lens[:i + 1]) for i in range(len(qo_lens))],
                                 dtype=torch.int32)
        assert shapes["q_nope"][0] == qo_indptr[-1].item()
        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        total_pages = sum(npages) + 3
        perm = torch.randperm(total_pages)
        indptr, indices = [0], []
        off = 0
        for n in npages:
            indices += perm[off:off + n].tolist()
            off += n
            indptr.append(len(indices))
        kv_indptr = torch.tensor(indptr, dtype=torch.int32)
        kv_indices = torch.tensor(indices, dtype=torch.int32)
        kv_len_arr = torch.tensor(kv_lens, dtype=torch.int32)

        assert shapes["ckv_cache"][0] == total_pages and shapes["ckv_cache"][1] == page_size
        ckv_cache = random_tensor(shapes["ckv_cache"], dtypes["ckv_cache"])
        kpe_cache = random_tensor(shapes["kpe_cache"], dtypes["kpe_cache"])

        input_groups.append([q_nope, q_pe, ckv_cache, kpe_cache,
                             qo_indptr, kv_indptr, kv_indices, kv_len_arr,
                             causal, sm_scale, return_lse, return_lse_base_on_e])
    return input_groups


def get_init_inputs():
    return []
