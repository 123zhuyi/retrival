import json
import math
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, lhs, m_indices, num_groups, out_features):
        torch.manual_seed(42)
        rows, in_features = lhs.shape
        if m_indices.shape != (rows,):
            raise ValueError("m_indices must contain one group id per lhs row")
        key = (num_groups, out_features, in_features, lhs.device, lhs.dtype)
        if key not in self._cache:
            self._cache.clear()
            generator = torch.Generator()
            generator.manual_seed(42)
            weight = torch.randn(
                (num_groups, out_features, in_features),
                generator=generator,
                dtype=torch.float32,
            ) / math.sqrt(in_features)
            self._cache[key] = weight.to(device=lhs.device, dtype=lhs.dtype)
        selected_weight = self._cache[key][m_indices.long()]
        return torch.bmm(selected_weight, lhs.unsqueeze(-1)).squeeze(-1)


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
        num_groups = specs["num_groups"]["value"]
        lhs = _random_tensor(specs["lhs"], 42 + case_index * 2)
        generator = torch.Generator()
        generator.manual_seed(43 + case_index * 2)
        m_indices = torch.randint(
            0,
            num_groups,
            tuple(specs["m_indices"]["shape"]),
            generator=generator,
            dtype=torch.int32,
        ).npu()
        input_groups.append(
            [lhs, m_indices, num_groups, specs["out_features"]["value"]]
        )
    return input_groups


def get_init_inputs():
    return []
