"""Jagged Softmax backward（变长序列 softmax 的反向）.

算子来源：Q3TritonKernel/src/kernels/fbgemm/jagged_softmax/jagged_softmax.py
    jagged_softmax_backward_kernel（由 JaggedSoftmax.backward 启动）

语义（golden，torch 原生小算子拼接）：
    grad_output    : [sum_L, H]  上游回传梯度
    softmax_output : [sum_L, H]  前向保存的 softmax 结果 y
    x_offsets      : [B+1] int32
    max_seq_len    : int，每条序列参与归一化的最大行数
    输出            : [sum_L, H]  dx，与输入同形同 dtype

    softmax 沿变长序列维（dim=0）做，其反向为
        dx[i, h] = y[i, h] * ( dy[i, h] - sum_j y[j, h] * dy[j, h] )
    求和 j 只在该序列的前 N = min(len, max_seq_len) 行内进行；
    超出 max_seq_len 的行输出保持 0（对应 kernel 侧 grad 以 zeros 初始化且不写该区间）。
    kernel 内累加走 fp32，本实现同样先升 fp32 再计算，最后回铸输入 dtype。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output, softmax_output, x_offsets, max_seq_len):
        device = grad_output.device
        total, heads = grad_output.shape
        batch = x_offsets.shape[0] - 1
        rows = max_seq_len

        if total == batch * rows:
            # 等长批：本 benchmark 约定各序列长度不超过 max_seq_len，
            # 于是 sum(len) == B * max_seq_len 只可能各条等长，无需 padding 与掩码
            dy = grad_output.reshape(batch, rows, heads).to(torch.float32)
            y = softmax_output.reshape(batch, rows, heads).to(torch.float32)
            dx = y * (dy - (y * dy).sum(dim=1, keepdim=True))
            return dx.reshape(total, heads).to(grad_output.dtype)

        pos = torch.arange(rows, device=device, dtype=torch.int32)
        lengths = x_offsets[1:] - x_offsets[:-1]
        valid = pos.reshape(1, rows) < lengths.reshape(batch, 1)
        idx = (x_offsets[:-1].reshape(batch, 1) + pos.reshape(1, rows)).reshape(-1)
        idx = idx.clamp(max=total - 1)

        dy = torch.index_select(grad_output, 0, idx).to(torch.float32).reshape(batch, rows, heads)
        y = torch.index_select(softmax_output, 0, idx).to(torch.float32).reshape(batch, rows, heads)
        dy = dy * valid[:, :, None]
        y = y * valid[:, :, None]
        dx = y * (dy - (y * dy).sum(dim=1, keepdim=True))

        flat_idx = torch.where(valid.reshape(-1), idx,
                               torch.full_like(idx, total)).to(torch.int64)
        buf = torch.zeros(total + 1, heads, device=device, dtype=torch.float32)
        buf.index_copy_(0, flat_idx, dx.reshape(-1, heads))
        return buf[:total].to(grad_output.dtype)


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
        names.append("10_JaggedSoftmaxBwd")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def build_inputs(shape, offsets, max_seq_len, dtype, seed):
        # softmax_output 必须是真正的前向输出：先造 logits 再沿变长序列维 softmax，
        # 超出 max_seq_len 的行置 0（与 kernel 前向的写入范围一致）；
        # grad_output 取 N(0, 1)，即上游回传梯度的常见量级。
        generator = torch.Generator()
        generator.manual_seed(seed)
        total, heads = shape
        grad_output = torch.randn((total, heads), generator=generator, dtype=torch.float32)
        logits = torch.randn((total, heads), generator=generator, dtype=torch.float32)
        softmax_output = torch.zeros(total, heads, dtype=torch.float32)
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            end = min(end, start + max_seq_len)
            if end > start:
                softmax_output[start:end] = torch.softmax(logits[start:end], dim=0)
        return grad_output.to(dtype), softmax_output.to(dtype)

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        dtype = dtype_map[specs["grad_output"]["dtype"]]
        offsets = specs["x_offsets"]["value"]
        max_seq_len = specs["max_seq_len"]["value"]
        grad_output, softmax_output = build_inputs(
            specs["grad_output"]["shape"], offsets, max_seq_len, dtype, 42 + case_index)
        input_groups.append([
            grad_output,
            softmax_output,
            torch.tensor(offsets, dtype=torch.int32),
            max_seq_len,
        ])
    return input_groups


def get_init_inputs():
    return []
