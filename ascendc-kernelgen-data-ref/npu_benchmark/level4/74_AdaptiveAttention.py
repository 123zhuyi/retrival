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

    def _layers(self, x, n_heads, n_kv_heads_options):
        d_model = x.shape[-1]
        head_dim = d_model // n_heads
        options = tuple(n_kv_heads_options)
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if any(n_heads % option != 0 for option in options):
            raise ValueError("each KV head option must divide n_heads")
        key = (d_model, n_heads, options, x.device, x.dtype)
        if key not in self._cache:
            self._cache.clear()
            rng_state = torch.get_rng_state()
            torch.manual_seed(42)
            router = nn.Sequential(
                nn.Linear(d_model, max(1, d_model // 4)),
                nn.ReLU(),
                nn.Linear(max(1, d_model // 4), len(options)),
            ).to(device=x.device, dtype=x.dtype)
            attention_layers = []
            for n_kv_heads in options:
                attention_layers.append((
                    nn.Linear(d_model, d_model, bias=False).to(
                        device=x.device, dtype=x.dtype
                    ),
                    nn.Linear(
                        d_model, n_kv_heads * head_dim, bias=False
                    ).to(device=x.device, dtype=x.dtype),
                    nn.Linear(
                        d_model, n_kv_heads * head_dim, bias=False
                    ).to(device=x.device, dtype=x.dtype),
                    nn.Linear(d_model, d_model, bias=False).to(
                        device=x.device, dtype=x.dtype
                    ),
                ))
            self._cache[key] = (router, tuple(attention_layers))
            torch.set_rng_state(rng_state)
        return self._cache[key]

    def forward(self, x, n_heads, n_kv_heads_options):
        torch.manual_seed(42)
        router, attention_layers = self._layers(
            x, n_heads, n_kv_heads_options
        )
        batch, sequence, d_model = x.shape
        head_dim = d_model // n_heads
        routing_logits = router(x.mean(dim=1))
        selected = F.one_hot(
            routing_logits.argmax(dim=-1), num_classes=len(attention_layers)
        ).to(dtype=x.dtype)
        candidates = []
        for n_kv_heads, layers in zip(n_kv_heads_options, attention_layers):
            q_proj, k_proj, v_proj, out_proj = layers
            q = q_proj(x).view(
                batch, sequence, n_heads, head_dim
            ).transpose(1, 2)
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
            output = output.transpose(1, 2).contiguous().view(
                batch, sequence, d_model
            )
            candidates.append(out_proj(output))
        stacked = torch.stack(candidates, dim=1)
        return (
            stacked * selected.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)


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
            specs["n_kv_heads_options"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
