"""Tests for the FusedRopeWithQkNormAndKvCacheUpdate custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.fused_rope_with_qk_norm_and_kv_cache_update import (
    FusedRopeWithQkNormAndKvCacheUpdateInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/FusedRopeWithQkNormAndKvCacheUpdate"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestFusedRopeWithQkNormAndKvCacheUpdateInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, FusedRopeWithQkNormAndKvCacheUpdateInputGenerator)

    def test_cache_position_within_cache_range(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        value_cache = result[names.index("value_cache")]
        cache_position = result[names.index("cache_position")]
        max_pos = value_cache.shape[2]
        assert cache_position.dtype == torch.int64
        assert cache_position.min().item() >= 0
        assert cache_position.max().item() < max_pos

    def test_cache_position_shape_dtype(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        cache_position = result[names.index("cache_position")]
        assert cache_position.shape == tuple(workload.inputs["cache_position"].shape)
        assert cache_position.dtype == torch.int64

    def test_reproducible(self):
        definition = _definition()
        workload = _first_workload()
        r1 = gen_inputs(definition, workload, "cpu", 123)
        r2 = gen_inputs(definition, workload, "cpu", 123)
        for a, b in zip(r1, r2):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            else:
                assert a == b
