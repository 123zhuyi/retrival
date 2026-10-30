import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, cache_seqlens, page_table, causal):
        torch.manual_seed(42)
        batch, query_length, n_heads, head_dim = q.shape
        _, page_size, n_kv_heads, _ = k_cache.shape
        if n_heads % n_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        offsets = torch.arange(page_size, device=q.device)
        logical_indices = (
            page_table.long().unsqueeze(-1) * page_size + offsets
        ).reshape(batch, -1)
        capacity = logical_indices.shape[1]
        flat_k = k_cache.reshape(-1, n_kv_heads, head_dim)
        flat_v = v_cache.reshape(-1, n_kv_heads, head_dim)
        k = flat_k[logical_indices].permute(0, 2, 1, 3).float()
        v = flat_v[logical_indices].permute(0, 2, 1, 3).float()
        repeats = n_heads // n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        q_heads = q.transpose(1, 2).float()
        scores = torch.matmul(q_heads, k.transpose(-2, -1))
        scores = scores / math.sqrt(head_dim)
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
        q = _random_tensor(specs["q"], 42 + case_index * 3)
        k_cache = _random_tensor(specs["k_cache"], 43 + case_index * 3)
        v_cache = _random_tensor(specs["v_cache"], 44 + case_index * 3)
        table_shape = tuple(specs["page_table"]["shape"])
        page_table = _page_table(
            table_shape, k_cache.shape[0], 142 + case_index
        )
        capacity = table_shape[1] * k_cache.shape[1]
        cache_seqlens = _cache_lengths(
            q.shape[0], capacity, q.shape[1], k_cache.shape[1], 242 + case_index
        )
        input_groups.append([
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            page_table,
            specs["causal"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
