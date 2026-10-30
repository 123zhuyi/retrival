"""Tests for the GroupedMatmulSwigluQuant custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.grouped_matmul_swiglu_quant import (
    GroupedMatmulSwigluQuantInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "GroupedMatmulSwigluQuant"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestGroupedMatmulSwigluQuantInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, GroupedMatmulSwigluQuantInputGenerator)

    def test_quant_tensors_are_small_integers(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        x, weight = result[0], result[1]
        assert x.dtype == torch.int8
        assert isinstance(weight, list)
        assert all(tensor.dtype == torch.int8 for tensor in weight)
        assert x.min().item() >= -5
        assert x.max().item() <= 4
        assert all(tensor.min().item() >= -5 for tensor in weight)
        assert all(tensor.max().item() <= 4 for tensor in weight)

    def test_scales_are_float32(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        weight_scale, x_scale = result[2], result[3]
        assert isinstance(weight_scale, list)
        assert all(tensor.dtype == torch.float32 for tensor in weight_scale)
        assert x_scale.dtype == torch.float32
        assert len(weight_scale) == workload.inputs["weight_scale"].shape[0]
        assert all(
            tensor.shape == tuple(workload.inputs["weight_scale"].shape[1:])
            for tensor in weight_scale
        )
        assert x_scale.shape == tuple(workload.inputs["x_scale"].shape)

    def test_group_list_is_non_decreasing(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        x, weight, _, _, group_list = result
        m = x.shape[0]
        e = len(weight)
        assert group_list.dtype == torch.int64
        assert group_list.shape == (e,)
        assert group_list[-1].item() == m
        assert torch.all(group_list[1:] >= group_list[:-1])

    def test_reproducible(self):
        definition = _definition()
        workload = _first_workload()
        r1 = gen_inputs(definition, workload, "cpu", 123)
        r2 = gen_inputs(definition, workload, "cpu", 123)
        for a, b in zip(r1, r2):
            if isinstance(a, list):
                assert all(torch.equal(x, y) for x, y in zip(a, b))
            else:
                assert torch.equal(a, b)

    def test_weight_is_int8_even_when_json_metadata_is_int32(self):
        definition = _definition()
        with open(DATA_DIR / "workload.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                wl = Workload(**json.loads(line))
                if wl.inputs["weight"].dtype == "int32":
                    result = gen_inputs(definition, wl, "cpu", 42)
                    assert all(tensor.dtype == torch.int8 for tensor in result[1])
                    return
        pytest.skip("No int32 weight workload found")
