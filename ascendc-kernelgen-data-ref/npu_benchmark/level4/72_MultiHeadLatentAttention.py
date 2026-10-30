import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        headdim_v,
        causal,
    ):
        torch.manual_seed(42)
        batch, query_length, _, qk_dim = q.shape
        _, page_size, _, cache_dim = kv_cache.shape
        if cache_dim != qk_dim or not 0 < headdim_v <= qk_dim:
            raise ValueError("cache and value dimensions are inconsistent")
        offsets = torch.arange(page_size, device=q.device)
        logical_indices = (
            block_table.long().unsqueeze(-1) * page_size + offsets
        ).reshape(batch, -1)
        capacity = logical_indices.shape[1]
        flat_cache = kv_cache.reshape(-1, cache_dim)
        selected = flat_cache[logical_indices]
        k = selected.unsqueeze(1).float()
        v = selected[..., :headdim_v].unsqueeze(1).float()
        q_heads = q.transpose(1, 2).float()
        scores = torch.matmul(q_heads, k.transpose(-2, -1))
        scores = scores / math.sqrt(qk_dim)
        positions = torch.arange(capacity, device=q.device)
        valid = positions.unsqueeze(0) < cache_seqlens.long().unsqueeze(1)
        allowed = valid.unsqueeze(1).expand(batch, query_length, capacity)
        if causal:
            query_positions = torch.arange(
                query_length, device=q.device
            ).view(1, query_length, 1)
            last_positions = (
                cache_seqlens.long().view(batch, 1, 1)
                - query_length
                + query_positions
            )
            allowed = allowed & (positions.view(1, 1, capacity) <= last_positions)
        scores = scores.masked_fill(
            (~allowed).unsqueeze(1), float("-inf")
        )
        output = torch.matmul(F.softmax(scores, dim=-1), v)
        return output.transpose(1, 2).to(dtype=q.dtype)


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


def _page_table(shape, num_blocks, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    rows, columns = shape
    table = [
        torch.randperm(num_blocks, generator=generator)[:columns]
        for _ in range(rows)
    ]
    return torch.stack(table).to(dtype=torch.int32).npu()


def _cache_lengths(batch, capacity, query_length, page_size, seed):
    values = []
    for row in range(batch):
        reduction = (seed * 7 + row * 3) % max(1, page_size)
        values.append(max(query_length, capacity - reduction))
    return torch.tensor(values, dtype=torch.int32).npu()


def get_input_groups():
    torch.manual_seed(42)
    input_groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = _case_specs(case)
        q = _random_tensor(specs["q"], 42 + case_index * 2)
        kv_cache = _random_tensor(specs["kv_cache"], 43 + case_index * 2)
        table_shape = tuple(specs["block_table"]["shape"])
        block_table = _page_table(
            table_shape, kv_cache.shape[0], 142 + case_index
        )
        capacity = table_shape[1] * kv_cache.shape[1]
        cache_seqlens = _cache_lengths(
            q.shape[0],
            capacity,
            q.shape[1],
            kv_cache.shape[1],
            242 + case_index,
        )
        input_groups.append([
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            specs["headdim_v"]["value"],
            specs["causal"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
