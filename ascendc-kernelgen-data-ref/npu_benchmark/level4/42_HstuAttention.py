"""HSTU Attention forward（生成式推荐 HSTU 的变长因果注意力，SiLU 取代 softmax）.

算子来源：Q3TritonKernel/src/kernels/recsys/triton_hstu_attention.py
    triton_hstu_attention_fwd(N, alpha, q, k, v, seq_offsets, num_targets,
                              max_attn_len, contextual_seq_len,
                              sort_by_length_indices, enable_tma, scaling_seqlen)

语义（golden，torch 原生小算子拼接；对应 num_targets=None、max_attn_len=0、
contextual_seq_len=0 的基础变体）：
    q / k     : [sum_L, H, D]   varlen 拼接的 query / key
    v         : [sum_L, H, Dv]
    seq_offsets: [B+1] int32/int64
    max_seq_len: int（同时作为打分缩放分母 scaling_seqlen）
    alpha     : float，打分缩放

    对序列 b 内局部位置 i、head h：
        scores[i, j] = (q[i, h] · k[j, h]) * alpha
        w[i, j]      = SiLU(scores[i, j]) / max_seq_len，且仅保留 j <= i（因果，含对角线）
        out[i, h]    = sum_j w[i, j] * v[j, h]
    与 softmax 注意力不同，HSTU 不做归一化，逐点 SiLU 后直接加权求和。
    kernel 内 tl.dot 为 bf16/fp16 输入 + fp32 累加；本实现统一先升 fp32 再计算，
    最后回铸 v.dtype。
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    # 单块打分张量的元素上限（2^28 个 fp32 = 1GB），决定 query 分块大小
    SCORE_BUDGET = 2 ** 28

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, seq_offsets, max_seq_len, alpha):
        device = q.device
        total, heads, dim = q.shape
        dim_v = v.shape[-1]
        batch = seq_offsets.shape[0] - 1
        rows = max_seq_len
        pos = torch.arange(rows, device=device, dtype=torch.int32)

        uniform = total == batch * rows
        if uniform:
            # 等长批：sum(len) == B * max_seq_len 且各 len <= max_seq_len ⇒ 各条等长，
            # q/k/v 本身即 [B, rows, H, *] 连续布局，免去 gather / 掩码 / 散射还原
            q_pad = q.reshape(batch, rows, heads, dim).to(torch.float32)
            k_pad = k.reshape(batch, rows, heads, dim).to(torch.float32)
            v_pad = v.reshape(batch, rows, heads, dim_v).to(torch.float32)
            valid = None
        else:
            lengths = seq_offsets[1:] - seq_offsets[:-1]
            valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
            gidx = (seq_offsets[:-1].reshape(batch, 1) + pos.reshape(1, rows))
            gidx = gidx.reshape(-1).clamp(max=total - 1).to(torch.int64)
            q_pad = torch.index_select(q, 0, gidx).to(torch.float32).reshape(batch, rows, heads, dim)
            k_pad = torch.index_select(k, 0, gidx).to(torch.float32).reshape(batch, rows, heads, dim)
            v_pad = torch.index_select(v, 0, gidx).to(torch.float32).reshape(batch, rows, heads, dim_v)

        # 按 query 分块：每块只取因果可见的前 end 个 key，打分张量为 [B, H, blk, end]，
        # 不再一次物化 [B, H, L, L]（用户序列到 32k 时单个 head 就要 4GB fp32）。
        # 数学上与整块计算等价；序列较短时只有一块，计算与整块完全相同。
        blk = max(128, min(rows, self.SCORE_BUDGET // max(1, batch * heads * rows)))
        out = torch.empty(batch, rows, heads, dim_v, device=device, dtype=torch.float32)
        for start in range(0, rows, blk):
            end = min(start + blk, rows)
            scores = torch.einsum("bqhd,bkhd->bhqk", q_pad[:, start:end], k_pad[:, :end]) * alpha
            weights = F.silu(scores) / rows
            keep = pos[start:end].reshape(-1, 1) >= pos[:end].reshape(1, -1)
            keep = keep.reshape(1, 1, end - start, end)
            if valid is None:
                weights = weights * keep
            else:
                weights = weights * keep * valid[:, :end].reshape(batch, 1, 1, end)
            out[:, start:end] = torch.einsum("bhqk,bkhd->bqhd", weights, v_pad[:, :end])
        out = out.reshape(batch * rows, heads, dim_v)

        if uniform:
            return out.to(v.dtype)
        flat_idx = torch.where(valid.reshape(-1), gidx, torch.full_like(gidx, total))
        buf = torch.zeros(total + 1, heads, dim_v, device=device, dtype=torch.float32)
        buf.index_copy_(0, flat_idx, out)
        return buf[:total].to(v.dtype)


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
        names.append("7_HstuAttention")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # q / k 取 N(0, 1)：HSTU 的 q/k 来自 LayerNorm 之后的 item embedding；
        # v 取 U(0, 1)：输出是 v 的加权和，v 若零均值会让输出集中在 0 附近，
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

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        dtype = dtype_map[specs["q"]["dtype"]]
        input_groups.append([
            random_tensor("q", specs["q"]["shape"], dtype, 42 + case_index * 3),
            random_tensor("k", specs["k"]["shape"], dtype, 43 + case_index * 3),
            random_tensor("v", specs["v"]["shape"], dtype, 44 + case_index * 3),
            torch.tensor(specs["seq_offsets"]["value"], dtype=torch.int64),
            specs["max_seq_len"]["value"],
            specs["alpha"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
