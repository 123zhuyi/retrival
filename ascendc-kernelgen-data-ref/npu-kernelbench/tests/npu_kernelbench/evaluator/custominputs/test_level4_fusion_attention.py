"""Tests for the FusionAttention custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.fusion_attention import (
    FusionAttentionInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "FusionAttention"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestFusionAttentionInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, FusionAttentionInputGenerator)

    def test_query_key_value_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for tensor_name in ("query", "key", "value"):
            tensor = result[names.index(tensor_name)]
            assert tuple(tensor.shape) == tuple(workload.inputs[tensor_name].shape)
            assert tensor.dtype == torch.float16

    def test_scalar_hyperparameters_passed_through(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        assert result[names.index("head_num")] == workload.inputs["head_num"].value
        assert result[names.index("input_layout")] == workload.inputs["input_layout"].value

    def test_uint8_atten_mask_workload_generates_ones(self):
        definition = _definition()
        workload = None
        with open(DATA_DIR / "workload.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                candidate = Workload(**json.loads(line))
                if candidate.uuid == "FusionAttention-058":
                    workload = candidate
                    break
        assert workload is not None

        result = gen_inputs(definition, workload, "cpu", 42)
        atten_mask = result[list(definition.inputs).index("atten_mask")]

        assert len(result) == len(definition.inputs)
        assert atten_mask.dtype == torch.uint8
        assert torch.all(atten_mask == 1)

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
