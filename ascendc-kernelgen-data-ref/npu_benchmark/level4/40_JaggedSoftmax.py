"""Jagged Softmax（FBGEMM 变长序列 softmax，jagged attention 的归一化环节）.

算子来源：Q3TritonKernel/src/kernels/fbgemm/jagged_softmax/jagged_softmax.py
    jagged_softmax_(x, x_offsets, max_seq_len)

语义（golden，torch 原生小算子拼接）：
    x         : [sum_L, H]   变长序列拼接后的打分（H 为并行的 head/特征列）
    x_offsets : [B+1] int32  每条序列在 sum_L 上的起止
    max_seq_len: int         每条序列参与归一化的最大行数
    输出       : [sum_L, H]   与 x 同形同 dtype

    对每条序列 b，记 N = min(offsets[b+1]-offsets[b], max_seq_len)：
        y[start : start+N] = softmax(x[start : start+N], dim=0)   # 沿变长序列维
        y[start+N : end]   = 0                                     # 超出 max_seq_len 的行保持 0
    归一化全程 fp32（与 kernel 的 tl.float32 累加一致）后回铸 x.dtype。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_offsets, max_seq_len):
        device = x.device
        total, heads = x.shape
        batch = x_offsets.shape[0] - 1
        # padding 行数直接取 max_seq_len（kernel 也只归一化前 max_seq_len 行），
        # 避免 max().item() 触发额外 reduce 与主机同步
        rows = max_seq_len

        if total == batch * rows:
            # 等长批：本 benchmark 约定各序列长度不超过 max_seq_len，
            # 于是 sum(len) == B * max_seq_len 只可能各条等长，
            # x 本身即 [B, rows, H] 连续布局，无需 padding / 掩码 / 散射还原
            padded = x.reshape(batch, rows, heads).to(torch.float32)
            return torch.softmax(padded, dim=1).reshape(total, heads).to(x.dtype)

        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = x_offsets[1:] - x_offsets[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        idx = x_offsets[:-1].reshape(batch, 1) + pos.reshape(1, rows)

        padded = torch.index_select(x, 0, idx.reshape(-1).clamp(max=total - 1))
        padded = padded.to(torch.float32).reshape(batch, rows, heads)
        padded = padded.masked_fill(~valid[:, :, None], float("-inf"))
        probs = torch.softmax(padded, dim=1)

        flat_idx = torch.where(valid.reshape(-1), idx.reshape(-1),
                               torch.full_like(idx.reshape(-1), total))
        buf = torch.zeros(total + 1, heads, device=device, dtype=torch.float32)
        buf.index_copy_(0, flat_idx.to(torch.int64), probs.reshape(-1, heads))
        return buf[:total].to(x.dtype)


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
        names.append("5_JaggedSoftmax")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(shape, dtype, seed):
        # 打分输入取 N(0, 1)：softmax 前的 logits 经缩放后即为该量级，
        # 既不会让 softmax 饱和成 one-hot，也不会退化成均匀分布。
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(dtype)

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        dtype = dtype_map[specs["x"]["dtype"]]
        input_groups.append([
            random_tensor(specs["x"]["shape"], dtype, 42 + case_index * 2),
            torch.tensor(specs["x_offsets"]["value"], dtype=torch.int32),
            specs["max_seq_len"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
