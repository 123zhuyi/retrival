"""Tests for the GroupedMatmulSwigluQuantV2 custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.grouped_matmul_swiglu_quant_v2 import (
    GroupedMatmulSwigluQuantV2InputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "GroupedMatmulSwigluQuantV2"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestGroupedMatmulSwigluQuantV2InputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, GroupedMatmulSwigluQuantV2InputGenerator)

    def test_x_and_weight_are_int8(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        x = result[names.index("x")]
        weight = result[names.index("weight")]
        assert x.dtype == torch.int8
        assert weight[0].dtype == torch.int8
        assert x.shape == tuple(workload.inputs["x"].shape)
        assert weight[0].shape == tuple(workload.inputs["weight"].shape)

    def test_weight_and_weight_scale_are_lists(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        weight = result[names.index("weight")]
        weight_scale = result[names.index("weight_scale")]
        assert isinstance(weight, list)
        assert isinstance(weight_scale, list)
        assert len(weight) == 1
        assert len(weight_scale) == 1

    def test_group_list_from_json_value(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        group_list = result[names.index("group_list")]
        expected_values = workload.inputs["group_list"].value
        assert group_list.dtype == torch.int64
        assert group_list.tolist() == expected_values

    def test_optional_tensors_are_none(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for name in ("smooth_scale", "weight_assist_matrix", "bias"):
            assert result[names.index(name)] is None

    def test_reproducible(self):
        definition = _definition()
        workload = _first_workload()
        r1 = gen_inputs(definition, workload, "cpu", 123)
        r2 = gen_inputs(definition, workload, "cpu", 123)
        for a, b in zip(r1, r2):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            elif isinstance(a, list):
                for t1, t2 in zip(a, b):
                    assert torch.equal(t1, t2)
            else:
                assert a == b
