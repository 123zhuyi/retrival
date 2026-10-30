"""Jagged-Dense Flash Attention forward（FBGEMM 推荐系统变长注意力）.

算子来源：Q3TritonKernel/src/kernels/fbgemm/jagged_dense_flash_attention/
          triton_jagged_dense_flash_attention.py
    jagged_dense_flash_attention_fwd(Q, K, V, attn_bias, jagged_offsets, max_seq_len, allow_tf32)

语义（golden，torch 原生小算子拼接）：
    Q            : [sum_L, D]   变长序列拼接的 query（jagged）
    K            : [B, D, T]    dense key（每个 batch 一份 T 个 slot）
    V            : [sum_L, D]   变长 value（jagged，与 Q 同形）
    attn_bias    : [B, N, T]    N = max_seq_len
    jagged_offsets: [B+1]       各 batch 在 sum_L 上的起止
    输出          : [B, T, D]

    对每个 batch b，记 L = min(offsets[b+1]-offsets[b], max_seq_len)：
        logits = Q[b 段] @ K[b]              # [L, T]
        logits = logits + attn_bias[b, :L]   # 加 bias
        probs  = softmax(logits, dim=0)      # 注意：沿 **jagged 序列维** 归一化
        out[b] = probs^T @ V[b 段]           # [T, D]
    L == 0 的 batch 输出保持 0（对应 kernel 的 early return）。

    kernel 内 tl.dot 走 fp32（allow_tf32=False），本实现同样全程 fp32。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, attn_bias, jagged_offsets, max_seq_len):
        device = q.device
        batch, dim, slots = k.shape
        total = q.shape[0]
        # padding 行数直接取 max_seq_len（kernel 也只处理前 max_seq_len 行），
        # 避免 max().item() 触发额外 reduce 与主机同步
        rows = max_seq_len

        if total == batch * rows:
            # 等长批：本 benchmark 约定各序列长度不超过 max_seq_len，
            # 于是 sum(len) == B * max_seq_len 只可能各条等长，q/v 本身即
            # [B, rows, D] 连续布局，无需 padding / 掩码 / 散射还原
            q_pad = q.reshape(batch, rows, dim).to(torch.float32)
            v_pad = v.reshape(batch, rows, dim).to(torch.float32)
            logits = torch.baddbmm(attn_bias[:, :rows, :].to(torch.float32),
                                   q_pad, k.to(torch.float32))
            probs = torch.softmax(logits, dim=1)
            return torch.bmm(probs.transpose(1, 2), v_pad).to(q.dtype)

        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = jagged_offsets[1:] - jagged_offsets[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)                  # [B, N]
        idx = (jagged_offsets[:-1].reshape(batch, 1) + pos.reshape(1, rows)).reshape(-1)
        idx = idx.clamp(max=total - 1)

        q_pad = torch.index_select(q, 0, idx).to(torch.float32).reshape(batch, rows, dim)
        v_pad = torch.index_select(v, 0, idx).to(torch.float32).reshape(batch, rows, dim)

        logits = torch.baddbmm(attn_bias[:, :rows, :].to(torch.float32),
                               q_pad, k.to(torch.float32))                        # [B, N, T]
        logits = logits.masked_fill(~valid[:, :, None], float("-inf"))
        probs = torch.softmax(logits, dim=1)
        return torch.bmm(probs.transpose(1, 2), v_pad).to(q.dtype)


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
        names.append("3_JaggedDenseFlashAttention")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # q / k / attn_bias 取 N(0, 1)，与推荐场景中归一化后的 embedding 量级一致；
        # v 取 U(0, 1)（与上游 test_jagged_dense_flash_attention.py 的 torch.rand 一致）：
        # 输出是 v 沿变长维的凸组合，v 若零均值会让输出集中在 0 附近，
        # 低精度输出的相对误差被数值抵消放大，与算子实现无关。
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
            random_tensor("q", specs["q"]["shape"], dtype, 42 + case_index * 4),
            random_tensor("k", specs["k"]["shape"], dtype, 43 + case_index * 4),
            random_tensor("v", specs["v"]["shape"], dtype, 44 + case_index * 4),
            random_tensor("attn_bias", specs["attn_bias"]["shape"], dtype, 45 + case_index * 4),
            torch.tensor(specs["jagged_offsets"]["value"], dtype=torch.int32),
            specs["max_seq_len"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
