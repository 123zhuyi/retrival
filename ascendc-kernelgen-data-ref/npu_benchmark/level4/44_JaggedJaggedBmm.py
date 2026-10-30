"""Jagged-Jagged BMM（变长注意力反向里的 X^T·Y 分批累乘）.

算子来源：Q3TritonKernel/src/kernels/fbgemm/jagged_jagged_bmm/jagged_jagged_bmm.py
    triton_jagged_jagged_bmm(x_t, y, x_offsets, N)

语义（golden，torch 原生小算子拼接）：
    x         : [sum_L, D]   变长序列拼接的左矩阵
    y         : [sum_L, T]   变长序列拼接的右矩阵（与 x 同一套 offsets）
    x_offsets : [B+1] int32
    N         : int，每条序列参与计算的最大行数（max_seq_len）
    输出       : [B, D, T]

    对每条序列 b，记 L = min(offsets[b+1]-offsets[b], N)：
        out[b] = x[start : start+L]^T @ y[start : start+L]
    即 jagged 版的 batched matmul（dense 语义为 [B, D, N] @ [B, N, T]）。
    kernel 内 tl.dot 为输入 dtype + fp32 累加；本实现统一先升 fp32 再 bmm，
    最后回铸 x.dtype。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, x_offsets, max_seq_len):
        device = x.device
        total, dim = x.shape
        cols = y.shape[1]
        batch = x_offsets.shape[0] - 1
        rows = max_seq_len

        if total == batch * rows:
            # 等长批：本 benchmark 约定各序列长度不超过 max_seq_len，
            # 于是 sum(len) == B * max_seq_len 只可能各条等长，
            # x / y 本身即 [B, rows, *] 连续布局，无需 padding 与掩码
            x_pad = x.reshape(batch, rows, dim).to(torch.float32)
            y_pad = y.reshape(batch, rows, cols).to(torch.float32)
            return torch.bmm(x_pad.transpose(1, 2), y_pad).to(x.dtype)

        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = x_offsets[1:] - x_offsets[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        idx = (x_offsets[:-1].reshape(batch, 1) + pos.reshape(1, rows)).reshape(-1)
        idx = idx.clamp(max=total - 1)

        x_pad = torch.index_select(x, 0, idx).to(torch.float32).reshape(batch, rows, dim)
        y_pad = torch.index_select(y, 0, idx).to(torch.float32).reshape(batch, rows, cols)
        x_pad = x_pad * valid[:, :, None]          # 越界行置 0，等价于不参与累乘
        return torch.bmm(x_pad.transpose(1, 2), y_pad).to(x.dtype)


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
        names.append("9_JaggedJaggedBmm")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # x 取 N(0, 1)（归一化后的激活/梯度量级）；
        # y 取 U(0, 1)：输出是长度 L 的累加和，两侧都零均值时结果会集中在 0 附近，
        # 低精度输出的相对误差被数值抵消放大到与算子实现无关的程度。
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape = tuple(shape)
        if name == "y":
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
        dtype = dtype_map[specs["x"]["dtype"]]
        input_groups.append([
            random_tensor("x", specs["x"]["shape"], dtype, 42 + case_index * 2),
            random_tensor("y", specs["y"]["shape"], dtype, 43 + case_index * 2),
            torch.tensor(specs["x_offsets"]["value"], dtype=torch.int32),
            specs["max_seq_len"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
