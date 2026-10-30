"""DSA Lightning Indexer forward（DeepSeek Sparse Attention 索引打分）.

算子来源：triton-ascend-kernels/src/triton_ascend_kernels/attention/dsa_lightning_indexer.py
    dsa_lightning_indexer_forward(q, q_s, k)

语义（golden，torch 原生小算子拼接）：
    index_score[b, m, n] = sum_h  weights[b, m, h] * relu( q[b, m, h, :] · k[b, n, :] )

    query   : [B, M, H, D]  bf16/fp16，M 为本次打分的 query token 数（decode 场景 M=1）
    weights : [B, M, H]     fp32，每个 head 的打分权重
    k_cache : [B, N, D]     bf16/fp16，N 为 KV cache 长度
    输出     : [B, M, N]     fp32，供后续 topk 选块使用

    kernel 内 tl.dot 为 bf16/fp16 输入 + fp32 累加，本实现统一先升 fp32 再 matmul，
    relu 与加权求和全程 fp32，累加精度不低于 kernel。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, weights, k_cache):
        b, m, h, d = query.shape
        n = k_cache.shape[1]
        q = query.to(torch.float32).reshape(b, m * h, d)
        k = k_cache.to(torch.float32).transpose(1, 2)
        scores = torch.matmul(q, k).reshape(b, m, h, n)
        scores = torch.relu(scores)
        return (scores * weights.to(torch.float32).unsqueeze(-1)).sum(dim=2)


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
        names.append("1_DsaLightningIndexer")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(spec, seed):
        # query / k_cache 取 N(0, 1)：indexer 的 q/k 来自 RMSNorm 之后的激活，
        # 量级接近单位方差；weights 取 U(-1, 1)，与算子实测分布一致。
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape = tuple(spec["shape"])
        dtype = dtype_map[spec["dtype"]]
        if spec["name"] == "weights":
            tensor = torch.empty(shape, dtype=torch.float32)
            tensor.uniform_(-1.0, 1.0, generator=generator)
        else:
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        return tensor.to(dtype=dtype)

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        input_groups.append([
            random_tensor(specs["query"], 42 + case_index * 3),
            random_tensor(specs["weights"], 43 + case_index * 3),
            random_tensor(specs["k_cache"], 44 + case_index * 3),
        ])
    return input_groups


def get_init_inputs():
    return []
