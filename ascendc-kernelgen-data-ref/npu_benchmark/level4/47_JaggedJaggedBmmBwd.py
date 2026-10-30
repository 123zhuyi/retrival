"""Jagged-Jagged BMM backward（变长注意力反向里 X^T·Y 的梯度）.

算子来源：Q3TritonKernel/src/kernels/fbgemm/jagged_jagged_bmm/jagged_jagged_bmm.py
    JaggedJaggedBmm.backward —— 由两次 triton_jagged_dense_bmm 组成
    （kernel 在 src/kernels/fbgemm/jagged_dense_bmm/jagged_dense_bmm.py）

语义（golden，torch 原生小算子拼接）：
    前向 Z[b] = X[b]^T @ Y[b]，X: [sum_L, D]，Y: [sum_L, T]，Z: [B, D, T]
    反向：
        dX[b] = Y[b] @ dZ[b]^T      # [L, T] @ [T, D] -> [L, D]
        dY[b] = X[b] @ dZ[b]        # [L, D] @ [D, T] -> [L, T]
    每条序列只取前 L = min(len, max_seq_len) 行参与，超出的行梯度保持 0
    （对应 kernel 侧输出以 zeros 初始化且不写该区间）。
    kernel 内 tl.dot 为输入 dtype + fp32 累加；本实现统一先升 fp32 再 bmm，
    最后回铸输入 dtype。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, x_offsets, max_seq_len, grad_output):
        device = x.device
        total, dim = x.shape
        cols = y.shape[1]
        batch = x_offsets.shape[0] - 1
        rows = max_seq_len
        grad_z = grad_output.to(torch.float32)

        if total == batch * rows:
            # 等长批：本 benchmark 约定各序列长度不超过 max_seq_len，
            # 于是 sum(len) == B * max_seq_len 只可能各条等长，无需 padding 与掩码
            x_pad = x.reshape(batch, rows, dim).to(torch.float32)
            y_pad = y.reshape(batch, rows, cols).to(torch.float32)
            grad_x = torch.bmm(y_pad, grad_z.transpose(1, 2)).reshape(total, dim)
            grad_y = torch.bmm(x_pad, grad_z).reshape(total, cols)
            return grad_x.to(x.dtype), grad_y.to(y.dtype)

        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = x_offsets[1:] - x_offsets[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        idx = (x_offsets[:-1].reshape(batch, 1) + pos.reshape(1, rows)).reshape(-1)
        idx = idx.clamp(max=total - 1)

        x_pad = torch.index_select(x, 0, idx).to(torch.float32).reshape(batch, rows, dim)
        y_pad = torch.index_select(y, 0, idx).to(torch.float32).reshape(batch, rows, cols)
        grad_x = torch.bmm(y_pad, grad_z.transpose(1, 2))
        grad_y = torch.bmm(x_pad, grad_z)

        flat_idx = torch.where(valid.reshape(-1), idx,
                               torch.full_like(idx, total)).to(torch.int64)
        buf_x = torch.zeros(total + 1, dim, device=device, dtype=torch.float32)
        buf_y = torch.zeros(total + 1, cols, device=device, dtype=torch.float32)
        buf_x.index_copy_(0, flat_idx, grad_x.reshape(-1, dim))
        buf_y.index_copy_(0, flat_idx, grad_y.reshape(-1, cols))
        return buf_x[:total].to(x.dtype), buf_y[:total].to(y.dtype)


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
        names.append("12_JaggedJaggedBmmBwd")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # x / grad_output 取 N(0, 1)（激活与上游回传梯度的常见量级）；
        # y 取 U(0, 1)：dX 是沿 T 维的累加和，两侧都零均值时结果集中在 0 附近，
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
            random_tensor("x", specs["x"]["shape"], dtype, 42 + case_index * 3),
            random_tensor("y", specs["y"]["shape"], dtype, 43 + case_index * 3),
            torch.tensor(specs["x_offsets"]["value"], dtype=torch.int32),
            specs["max_seq_len"]["value"],
            random_tensor("grad_output", specs["grad_output"]["shape"], dtype, 44 + case_index * 3),
        ])
    return input_groups


def get_init_inputs():
    return []
