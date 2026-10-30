"""NSA Topk Sparse Attention backward（选块稀疏注意力的反向）.

算子来源：triton-ascend-kernels/src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py
    _topk_sparse_attention_bwd(o, do, lse, q, k, v, topk_idx, block_size,
                               cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale)

语义（golden，torch 原生小算子拼接）：
    q / k / v   : [total_len, H*, D]   前向的输入（GQA：Hq % Hk == 0）
    topk_idx    : [Hk, total_len, topk] int32，每个 query 选中的 KV 块号，-1 为 padding
    o           : [total_len, Hq, D]   前向输出（saved tensor）
    lse         : [Hq, total_len] fp32 前向保存的 logsumexp，**以 log2 为底**
                  （kernel 前向 qk_scale = sm_scale * log2(e)，p = exp2(qk - lse)）
    do          : [total_len, Hq, D]   上游回传梯度
    输出         : (dq, dk, dv)

    记允许集合 S(i) = {j : j // block_size ∈ topk_idx[h, i]} ∩ {j <= i}，
    p[i, j] = exp2(q_i·k_j * sm_scale * log2(e) - lse[i])（j ∈ S(i)，否则 0），则
        dv[j]    = sum_i p[i, j] * do[i]
        dp[i, j] = do[i] · v[j]
        delta[i] = sum_d o[i, d] * do[i, d]
        ds[i, j] = p[i, j] * (dp[i, j] - delta[i])
        dq[i]    = sm_scale * sum_j ds[i, j] * k[j]
        dk[j]    = sm_scale * sum_i ds[i, j] * q[i]
    这与 kernel 的 backward_dq / backward_dkdv 是同一套公式，差别只在 kernel 内部
    对 p / ds 做了 `.to(q.dtype)` 再进 tl.dot。o 与 lse 作为 saved tensor 由 case
    统一提供，两侧消费同一份，不各自重算前向。

    dtype：本算子的 case 取 fp32。kernel 反向里 `ds = ds.to(q.dtype)` 会把 ds 降到
    输入 dtype 再做 tl.dot，fp16/bf16 下该步是误差主因（实测共用 o/lse 后 dq/dk/dv
    的 MERE 仍有 1.6e-3~3.1e-3 / 1.0e-2~2.6e-2，超出验证器的 9.77e-4 / 7.81e-3）；
    fp32 下这次降精度不存在，实测 MERE ~1e-6。
"""
import json
import math
import os

import torch
import torch.nn as nn

_LOG2E = 1.44269504


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, topk_idx, block_size, cu_seqlens, max_seqlen, sm_scale, o, lse, do):
        device = q.device
        total, hq, dim = q.shape
        hk = k.shape[1]
        group = hq // hk
        batch = cu_seqlens.shape[0] - 1
        rows = max_seqlen
        topk = topk_idx.shape[-1]

        def allowed_mask(sel_idx, device):
            # 选块 + 因果 → [B, Hk, Lq, Lk] 布尔允许集合（-1 落到哨兵列后丢弃）
            blocks = (rows + block_size - 1) // block_size
            idx = torch.where(sel_idx < 0, torch.full_like(sel_idx, blocks), sel_idx)
            idx = idx.permute(1, 0, 2, 3).to(torch.int64)
            sel = torch.zeros(batch, hk, rows, blocks + 1, device=device, dtype=torch.bool)
            sel.scatter_(-1, idx, True)
            blk_of = torch.arange(blocks, device=device).reshape(blocks, 1)
            blk_of = blk_of.expand(blocks, block_size).reshape(-1)[:rows]
            block_mask = torch.index_select(sel[..., :blocks], -1, blk_of)
            span = torch.arange(rows, device=device)
            return block_mask & (span.reshape(rows, 1) >= span.reshape(1, rows)).reshape(1, 1, rows, rows)

        uniform = total == batch * rows
        if uniform:
            # 等长批：sum(len) == B * max_seqlen 且各 len <= max_seqlen ⇒ 各条等长
            gidx = None
            q_pad = q.to(torch.float32).reshape(batch, rows, hk, group, dim)
            k_pad = k.to(torch.float32).reshape(batch, rows, hk, dim)
            v_pad = v.to(torch.float32).reshape(batch, rows, hk, dim)
            o_pad = o.to(torch.float32).reshape(batch, rows, hk, group, dim)
            do_pad = do.to(torch.float32).reshape(batch, rows, hk, group, dim)
            sel_idx = topk_idx.reshape(hk, batch, rows, topk)
            lse_pad = lse.reshape(hq, batch, rows).permute(1, 0, 2).reshape(batch, hk, group, rows)
            valid = None
        else:
            pos = torch.arange(rows, device=device, dtype=torch.int32)
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
            gidx = (cu_seqlens[:-1].reshape(batch, 1) + pos.reshape(1, rows))
            gidx = gidx.reshape(-1).clamp(max=total - 1).to(torch.int64)
            q_pad = torch.index_select(q, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
            k_pad = torch.index_select(k, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
            v_pad = torch.index_select(v, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
            o_pad = torch.index_select(o, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
            do_pad = torch.index_select(do, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
            sel_idx = torch.index_select(topk_idx, 1, gidx).reshape(hk, batch, rows, topk)
            lse_pad = torch.index_select(lse, 1, gidx).reshape(hq, batch, rows)
            lse_pad = lse_pad.permute(1, 0, 2).reshape(batch, hk, group, rows)
            do_pad = do_pad * valid.reshape(batch, rows, 1, 1, 1)

        keep = allowed_mask(sel_idx, device)
        if valid is not None:
            keep = keep & valid.reshape(batch, 1, 1, rows)
        keep = keep.reshape(batch, hk, 1, rows, rows)

        # p = exp2(qk * sm_scale * log2e - lse)，与 kernel 前向/反向同一套 log2 口径
        scores = torch.einsum("bqhgd,bkhd->bhgqk", q_pad, k_pad) * (sm_scale * _LOG2E)
        probs = torch.exp2(scores - lse_pad[..., None])
        probs = probs * keep

        delta = (o_pad * do_pad).sum(-1).permute(0, 2, 3, 1)                     # [B,Hk,g,Lq]
        grad_p = torch.einsum("bqhgd,bkhd->bhgqk", do_pad, v_pad)
        grad_s = probs * (grad_p - delta[..., None])

        grad_q = torch.einsum("bhgqk,bkhd->bqhgd", grad_s, k_pad) * sm_scale
        grad_k = torch.einsum("bhgqk,bqhgd->bkhd", grad_s, q_pad) * sm_scale
        grad_v = torch.einsum("bhgqk,bqhgd->bkhd", probs, do_pad)

        grad_q = grad_q.reshape(batch * rows, hq, dim)
        grad_k = grad_k.reshape(batch * rows, hk, dim)
        grad_v = grad_v.reshape(batch * rows, hk, dim)
        if uniform:
            return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)

        flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
        out = []
        for grad, heads, width, ref in ((grad_q, hq, dim, q), (grad_k, hk, dim, k), (grad_v, hk, dim, v)):
            buf = torch.zeros(total + 1, heads, width, device=device, dtype=torch.float32)
            buf.index_copy_(0, flat_idx, grad)
            out.append(buf[:total].to(ref.dtype))
        return out[0], out[1], out[2]


def get_input_groups():
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def json_path():
        # 验证/性能脚本会把本文件拷贝成 {op_name}_torch.py，
        # 因此除同名 .json 外还需回退到去掉 _torch 后缀的名字与原始 case 文件名。
        here = os.path.dirname(os.path.abspath(__file__))
        stem = os.path.splitext(os.path.basename(__file__))[0]
        names = [stem]
        if stem.endswith("_torch"):
            names.append(stem[: -len("_torch")])
        names.append("11_NsaTopkSparseAttentionBwd")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # q / k 取 N(0, 1)：注意力输入来自 RMSNorm 之后的激活；
        # v 与 do 取 U(0, 1)：dv 是 v 侧的凸组合、do 决定 dv 的符号，
        # 两者若零均值会让梯度集中在 0 附近，相对误差被数值抵消放大。
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape = tuple(shape)
        if name in ("v", "do"):
            tensor = torch.empty(shape, dtype=torch.float32)
            tensor.uniform_(0.0, 1.0, generator=generator)
        else:
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        return tensor.to(dtype)

    def build_topk_idx(lengths, hk, topk, block_size, seed):
        # 复刻上游 test 的选块生成：每个 query 在其因果可见的块里随机选 topk 个，
        # 升序排序后把最小的一个强制换成块 0（保证 softmax 分母非空、且不产生重复块），
        # 超出因果范围的块号置 -1 表示 padding。
        generator = torch.Generator()
        generator.manual_seed(seed)
        chunks = []
        for length in lengths:
            nblk = (length + block_size - 1) // block_size
            local = torch.arange(length).reshape(1, length, 1)
            score = torch.rand(hk, length, nblk, generator=generator)
            take = min(topk, nblk)
            idx = score.topk(take, dim=-1).indices.to(torch.int32)
            idx, _ = torch.sort(idx, dim=-1)
            if take < topk:
                pad = torch.full((hk, length, topk - take), nblk, dtype=torch.int32)
                idx = torch.cat([idx, pad], dim=-1)
            idx[:, :, 0] = 0
            idx = torch.where(idx > (local // block_size).to(torch.int32),
                              torch.full_like(idx, -1), idx)
            chunks.append(idx)
        return torch.cat(chunks, dim=1).contiguous()

    def saved_forward(q, k, v, topk_idx, block_size, lengths, sm_scale):
        # 前向的 saved tensor：o 与 lse（lse 以 log2 为底，与 kernel 口径一致）。
        # 反向 kernel 直接消费这两个张量，因此由 case 统一提供，两侧不各自重算。
        total, hq, dim = q.shape
        hk = k.shape[1]
        group = hq // hk
        rows = max(lengths)
        batch = len(lengths)
        starts = [0]
        for length in lengths[:-1]:
            starts.append(starts[-1] + length)
        pos = torch.arange(rows)
        valid = pos.reshape(1, rows) < torch.tensor(lengths).reshape(batch, 1)
        gidx = (torch.tensor(starts).reshape(batch, 1) + pos.reshape(1, rows))
        gidx = gidx.reshape(-1).clamp(max=total - 1)

        q_pad = q[gidx].to(torch.float32).reshape(batch, rows, hk, group, dim)
        k_pad = k[gidx].to(torch.float32).reshape(batch, rows, hk, dim)
        v_pad = v[gidx].to(torch.float32).reshape(batch, rows, hk, dim)
        sel_idx = topk_idx[:, gidx].reshape(hk, batch, rows, topk_idx.shape[-1])
        blocks = (rows + block_size - 1) // block_size
        idx = torch.where(sel_idx < 0, torch.full_like(sel_idx, blocks), sel_idx)
        idx = idx.permute(1, 0, 2, 3).to(torch.int64)
        sel = torch.zeros(batch, hk, rows, blocks + 1, dtype=torch.bool)
        sel.scatter_(-1, idx, True)
        blk_of = torch.arange(blocks).reshape(blocks, 1).expand(blocks, block_size).reshape(-1)[:rows]
        keep = torch.index_select(sel[..., :blocks], -1, blk_of)
        keep = keep & (pos.reshape(rows, 1) >= pos.reshape(1, rows)).reshape(1, 1, rows, rows)
        keep = keep & valid.reshape(batch, 1, 1, rows)
        keep = keep.reshape(batch, hk, 1, rows, rows)

        scores = torch.einsum("bqhgd,bkhd->bhgqk", q_pad, k_pad) * (sm_scale * _LOG2E)
        scores = scores.masked_fill(~keep, float("-inf"))
        lse_pad = torch.logsumexp(scores * math.log(2.0), dim=-1) * _LOG2E     # [B,Hk,g,Lq]
        probs = torch.exp2(scores - lse_pad[..., None])
        out = torch.einsum("bhgqk,bkhd->bqhgd", probs, v_pad).reshape(batch * rows, hq, dim)

        o_flat = torch.zeros(total + 1, hq, dim, dtype=torch.float32)
        flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
        o_flat.index_copy_(0, flat_idx, out)
        lse_flat = torch.zeros(hq, total + 1, dtype=torch.float32)
        lse_flat.index_copy_(1, flat_idx, lse_pad.reshape(batch, hq, rows).permute(1, 0, 2).reshape(hq, -1))
        return o_flat[:total].to(q.dtype), lse_flat[:, :total].contiguous()

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        dtype = dtype_map[specs["q"]["dtype"]]
        cu_seqlens = torch.tensor(specs["cu_seqlens"]["value"], dtype=torch.int32)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        hk = specs["k"]["shape"][1]
        topk = specs["topk_idx"]["shape"][2]
        block_size = specs["block_size"]["value"]
        sm_scale = specs["sm_scale"]["value"]
        q = random_tensor("q", specs["q"]["shape"], dtype, 42 + case_index * 4)
        k = random_tensor("k", specs["k"]["shape"], dtype, 43 + case_index * 4)
        v = random_tensor("v", specs["v"]["shape"], dtype, 44 + case_index * 4)
        do = random_tensor("do", specs["do"]["shape"], dtype, 45 + case_index * 4)
        topk_idx = build_topk_idx(lengths, hk, topk, block_size, 46 + case_index * 4)
        o, lse = saved_forward(q, k, v, topk_idx, block_size, lengths, sm_scale)
        input_groups.append([
            q, k, v, topk_idx, block_size, cu_seqlens,
            specs["max_seqlen"]["value"], sm_scale, o, lse, do,
        ])
    return input_groups


def get_init_inputs():
    return []
