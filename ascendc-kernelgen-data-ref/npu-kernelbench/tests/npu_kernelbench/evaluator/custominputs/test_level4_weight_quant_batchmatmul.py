"""Tests for the WeightQuantBatchmatmul custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.weight_quant_batchmatmul import (
    WeightQuantBatchmatmulInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "WeightQuantBatchmatmul"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestWeightQuantBatchmatmulInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, WeightQuantBatchmatmulInputGenerator)

    def test_weight_is_int8_narrow_range(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        weight = result[names.index("weight")]
        assert weight.dtype == torch.int8
        assert weight.min().item() >= -8
        assert weight.max().item() < 8
        assert weight.shape == tuple(workload.inputs["weight"].shape)

    def test_optional_tensors_default_to_none(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for name in ("antiquant_offset", "quant_scale", "quant_offset", "bias"):
            assert result[names.index(name)] is None

    def test_antiquant_scale_defaults_to_ones_when_absent(self):
        definition = _definition()
        workload = _first_workload()
        # Remove antiquant_scale from the workload to exercise the default path.
        del workload.inputs["antiquant_scale"]
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        scale = result[names.index("antiquant_scale")]
        assert scale.shape == (1,)
        assert scale.dtype == torch.float16
        assert scale.item() == 1.0

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
