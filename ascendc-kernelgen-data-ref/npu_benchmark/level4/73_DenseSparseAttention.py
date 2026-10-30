import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, kv_cache, indices, headdim_v):
        torch.manual_seed(42)
        qk_dim = q.shape[-1]
        if kv_cache.shape[-1] != qk_dim or not 0 < headdim_v <= qk_dim:
            raise ValueError("cache and value dimensions are inconsistent")
        flat_cache = kv_cache.reshape(-1, qk_dim)
        selected = flat_cache[indices.long()]
        selected_k = selected.float()
        selected_v = selected[..., :headdim_v].float()
        scores = torch.einsum(
            "bqhd,bqkd->bhqk", q.float(), selected_k
        ) / math.sqrt(qk_dim)
        weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bhqk,bqkv->bqhv", weights, selected_v)
        return output.to(dtype=q.dtype)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    dtype = _DTYPE_MAP[spec["dtype"]]
    if torch.rand(1, generator=generator).item() < 0.5:
        mean = torch.rand(1, generator=generator).item() * 10.0 - 5.0
        std = torch.rand(1, generator=generator).item() * 1.9 + 0.1
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        tensor = tensor * std + mean
    else:
        tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
        tensor = tensor * 10.0 - 5.0
    return tensor.to(dtype=dtype).npu()


def _load_cases():
    json_path = os.path.splitext(__file__)[0] + ".json"
    with open(json_path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def _case_specs(case):
    return {item["name"]: item for item in case["inputs"]}


def get_input_groups():
    torch.manual_seed(42)
    input_groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = _case_specs(case)
        q = _random_tensor(specs["q"], 42 + case_index * 2)
        kv_cache = _random_tensor(specs["kv_cache"], 43 + case_index * 2)
        generator = torch.Generator()
        generator.manual_seed(142 + case_index)
        indices = torch.randint(
            0,
            kv_cache.shape[0] * kv_cache.shape[1],
            tuple(specs["indices"]["shape"]),
            generator=generator,
            dtype=torch.int32,
        ).npu()
        input_groups.append([
            q,
            kv_cache,
            indices,
            specs["headdim_v"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
