"""NSA Topk Sparse Attention forward（DeepSeek Native Sparse Attention 的选块注意力）.

算子来源：triton-ascend-kernels/src/triton_ascend_kernels/attention/nsa_topk_sparse_attention.py
    _topk_sparse_attention_fwd(q, k, v, topk_idx, block_size,
                               cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale)

语义（golden，torch 原生小算子拼接）：
    q         : [total_len, Hq, D]   varlen 拼接的 query
    k / v     : [total_len, Hk, D]   varlen 拼接的 key/value（GQA：Hq % Hk == 0）
    topk_idx  : [Hk, total_len, topk] int32，每个 query 选中的 KV 块号，-1 表示 padding
    block_size: KV 分块大小（32/64/128/256）
    cu_seqlens: [B+1] int32
    max_seqlen: int，批内最长序列长度（形状提示，等价 varlen flash-attn 的 max_seqlen）
    输出       : [total_len, Hq, D]

    对序列 b 内的 query 局部位置 i、kv head h：
        允许的 key 局部位置 j 需同时满足
            (a) j // block_size ∈ topk_idx[h, 全局 query 号]（-1 不计）
            (b) j <= i                    （因果）
        scores = q·k * sm_scale，在允许集合上做 softmax，再与 v 加权求和。
    Hq // Hk 个 query head 共享同一个 kv head 的选块结果。
    kernel 内 tl.dot 为 bf16/fp16 输入 + fp32 累加、online softmax；
    本实现统一先升 fp32 再计算，最后回铸 q.dtype。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, topk_idx, block_size, cu_seqlens, max_seqlen, sm_scale):
        device = q.device
        total, hq, dim = q.shape
        hk = k.shape[1]
        group = hq // hk
        batch = cu_seqlens.shape[0] - 1
        rows = max_seqlen
        blocks = (rows + block_size - 1) // block_size
        topk = topk_idx.shape[-1]
        pos = torch.arange(rows, device=device, dtype=torch.int32)

        uniform = total == batch * rows
        if uniform:
            # 等长批：sum(len) == B * max_seqlen 且各 len <= max_seqlen ⇒ 各条等长，
            # q/k/v/topk_idx 本身即按 [B, rows, ...] 连续排布，免去 gather / 掩码 / 散射
            q_pad = q.reshape(batch, rows, hk, group, dim).to(torch.float32)
            k_pad = k.reshape(batch, rows, hk, dim).to(torch.float32)
            v_pad = v.reshape(batch, rows, hk, dim).to(torch.float32)
            sel_idx = topk_idx.reshape(hk, batch, rows, topk)
            valid = None
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)               # [B, L]
            gidx = (cu_seqlens[:-1].reshape(batch, 1) + pos.reshape(1, rows))
            gidx = gidx.reshape(-1).clamp(max=total - 1).to(torch.int64)
            q_pad = torch.index_select(q, 0, gidx).to(torch.float32).reshape(batch, rows, hk, group, dim)
            k_pad = torch.index_select(k, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
            v_pad = torch.index_select(v, 0, gidx).to(torch.float32).reshape(batch, rows, hk, dim)
            sel_idx = torch.index_select(topk_idx, 1, gidx).reshape(hk, batch, rows, topk)

        # 选块表 → [B, Hk, L, blocks] 的布尔选择矩阵（-1 落到哨兵列后丢弃）。
        # 直接在 [B, Hk, ...] 布局上 scatter，避免事后对 [Hk, B, L, L] 大布尔张量做 permute；
        # 块号→列号的展开用 index_select（arange+expand 生成映射表），
        # 比 repeat_interleave 少一次大张量 gather，也避开整除落到 AiCpu。
        sel_idx = torch.where(sel_idx < 0, torch.full_like(sel_idx, blocks), sel_idx)
        sel_idx = sel_idx.permute(1, 0, 2, 3).to(torch.int64)
        sel = torch.zeros(batch, hk, rows, blocks + 1, device=device, dtype=torch.bool)
        sel.scatter_(-1, sel_idx, True)
        blk_of = torch.arange(blocks, device=device).reshape(blocks, 1)
        blk_of = blk_of.expand(blocks, block_size).reshape(-1)[:rows]
        block_mask = torch.index_select(sel[..., :blocks], -1, blk_of)               # [B, Hk, Lq, Lk]

        causal = pos.reshape(rows, 1) >= pos.reshape(1, rows)                        # [Lq, Lk] j <= i
        keep = block_mask & causal.reshape(1, 1, rows, rows)
        if valid is not None:
            keep = keep & valid.reshape(batch, 1, 1, rows)
        keep = keep.reshape(batch, hk, 1, rows, rows)

        scores = torch.einsum("bqhgd,bkhd->bhgqk", q_pad, k_pad) * sm_scale
        scores = scores.masked_fill(~keep, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhgqk,bkhd->bqhgd", probs, v_pad).reshape(batch * rows, hq, dim)

        if uniform:
            return out.to(q.dtype)
        flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
        buf = torch.zeros(total + 1, hq, dim, device=device, dtype=torch.float32)
        buf.index_copy_(0, flat_idx, out)
        return buf[:total].to(q.dtype)


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
        names.append("6_NsaTopkSparseAttention")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # q / k 取 N(0, 1)：注意力输入来自 RMSNorm 之后的激活；
        # v 取 U(0, 1)：输出是 v 的凸组合，v 若零均值会让输出集中在 0 附近，
        # 低精度输出的相对误差被数值抵消放大到与算子实现无关的程度。
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape = tuple(shape)
        if name == "v":
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
        input_groups.append([
            random_tensor("q", specs["q"]["shape"], dtype, 42 + case_index * 3),
            random_tensor("k", specs["k"]["shape"], dtype, 43 + case_index * 3),
            random_tensor("v", specs["v"]["shape"], dtype, 44 + case_index * 3),
            build_topk_idx(lengths, hk, topk, block_size, 45 + case_index * 3),
            block_size,
            cu_seqlens,
            specs["max_seqlen"]["value"],
            specs["sm_scale"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
