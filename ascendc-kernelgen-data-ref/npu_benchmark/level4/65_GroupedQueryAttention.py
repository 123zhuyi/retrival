import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def _layers(self, x, n_heads, n_kv_heads):
        d_model = x.shape[-1]
        if d_model % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError("head counts must divide d_model and each other")
        head_dim = d_model // n_heads
        key = (d_model, n_heads, n_kv_heads, x.device, x.dtype)
        if key not in self._cache:
            self._cache.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            self._cache[key] = (
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
                nn.Linear(d_model, n_kv_heads * head_dim, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
                nn.Linear(d_model, n_kv_heads * head_dim, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
                nn.Linear(d_model, d_model, bias=False).to(
                    device=x.device, dtype=x.dtype
                ),
            )
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def forward(self, x, n_heads, n_kv_heads):
        torch.manual_seed(42)
        q_proj, k_proj, v_proj, out_proj = self._layers(
            x, n_heads, n_kv_heads
        )
        batch, sequence, d_model = x.shape
        head_dim = d_model // n_heads
        q = q_proj(x).view(batch, sequence, n_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(
            batch, sequence, n_kv_heads, head_dim
        ).transpose(1, 2)
        v = v_proj(x).view(
            batch, sequence, n_kv_heads, head_dim
        ).transpose(1, 2)
        repeats = n_heads // n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        weights = F.softmax(scores / math.sqrt(head_dim), dim=-1)
        output = torch.matmul(weights, v.float()).to(dtype=x.dtype)
        output = output.transpose(1, 2).contiguous().view(batch, sequence, d_model)
        return out_proj(output)


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
        input_groups.append([
            _random_tensor(specs["x"], 42 + case_index),
            specs["n_heads"]["value"],
            specs["n_kv_heads"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
