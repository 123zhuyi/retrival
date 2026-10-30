"""Chunk Local Cumsum（线性注意力/门控 delta rule 里的分块门控累加）.

算子来源：triton-ascend-kernels/src/triton_ascend_kernels/attention/fla/cumsum_fla.py
    chunk_local_cumsum_fla(g, chunk_size, reverse, scale, cu_seqlens,
                           chunk_indices_out, head_first, output_dtype)

语义（golden，torch 原生小算子拼接）：
    g          : [B, T, H]  门控张量（log 空间），H 为 head 数
    chunk_size : 分块长度，累加只在块内进行（跨块不累加）
    reverse    : False 取块内前缀和；True 取块内后缀和
    scale      : 非 None 时对结果整体缩放
    输出        : [B, T, H] fp32

    对每个 chunk [s, e)（e = min(s + chunk_size, T)）：
        reverse=False: out[:, s+i] = sum_{j<=i} g[:, s+j]
        reverse=True : out[:, s+i] = sum_{j>=i} g[:, s+j]
    kernel 内累加走 fp32；本实现同样先升 fp32 再 cumsum，尾块以补零对齐
    （补零不影响块内前缀/后缀和）。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, chunk_size, reverse, scale):
        batch, seqlen, heads = g.shape
        chunks = (seqlen + chunk_size - 1) // chunk_size
        padded = chunks * chunk_size

        buf = torch.zeros(batch, padded, heads, device=g.device, dtype=torch.float32)
        buf[:, :seqlen] = g.to(torch.float32)
        blocks = buf.reshape(batch, chunks, chunk_size, heads)

        out = blocks.cumsum(dim=2)
        if reverse:
            # 块内后缀和：total - 前缀和 + 当前元素
            out = blocks.sum(dim=2, keepdim=True) - out + blocks
        if scale is not None:
            out = out * scale
        return out.reshape(batch, padded, heads)[:, :seqlen]


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
        names.append("8_ChunkLocalCumsum")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(shape, dtype, seed):
        # 门控 g 取 logsigmoid(U(0,1))：与 gated delta rule / GLA 中
        # g = F.logsigmoid(gate) 的真实取值范围一致（负值、量级 <1），
        # 块内累加不会溢出低精度表示范围。
        generator = torch.Generator()
        generator.manual_seed(seed)
        raw = torch.empty(tuple(shape), dtype=torch.float32)
        raw.uniform_(0.0, 1.0, generator=generator)
        return torch.nn.functional.logsigmoid(raw).to(dtype)

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        input_groups.append([
            random_tensor(specs["g"]["shape"], dtype_map[specs["g"]["dtype"]], 42 + case_index),
            specs["chunk_size"]["value"],
            bool(specs["reverse"]["value"]),
            specs["scale"]["value"] if specs["scale"]["value"] != 0 else None,
        ])
    return input_groups


def get_init_inputs():
    return []
