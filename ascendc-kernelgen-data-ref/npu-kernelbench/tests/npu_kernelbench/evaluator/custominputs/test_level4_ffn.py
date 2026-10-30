"""Tests for the FFN custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.dtypes import dtype_str_to_torch_dtype
from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.ffn import FFNInputGenerator
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "FFN"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _find_workload(predicate):
    with open(DATA_DIR / "workload.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            wl = Workload(**json.loads(line))
            if predicate(wl):
                return wl
    raise RuntimeError("No matching workload found")


class TestFFNInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, FFNInputGenerator)

    def test_core_inputs_shapes_and_dtypes(self):
        definition = _definition()
        workload = _find_workload(
            lambda w: set(w.inputs.keys()) == {"x", "weight1", "weight2", "activation", "inner_precise"}
        )
        result = gen_inputs(definition, workload, "cpu", 42)
        assert len(result) == len(definition.inputs)
        assert result[0].shape == tuple(workload.inputs["x"].shape)
        assert result[1].shape == tuple(workload.inputs["weight1"].shape)
        assert result[2].shape == tuple(workload.inputs["weight2"].shape)
        assert result[3] == workload.inputs["activation"].value
        names = list(definition.inputs)
        assert result[names.index("expert_tokens")] is None
        assert result[names.index("expert_tokens_index")] is None
        assert result[names.index("inner_precise")] == workload.inputs[
            "inner_precise"
        ].value
        for name, value in zip(names, result):
            if name not in workload.inputs:
                assert value is None

    def test_expert_tokens_passed_through(self):
        definition = _definition()
        workload = _find_workload(lambda w: "expert_tokens" in w.inputs)
        result = gen_inputs(definition, workload, "cpu", 42)
        expert_tokens = result[list(definition.inputs.keys()).index("expert_tokens")]
        assert expert_tokens == workload.inputs["expert_tokens"].value
        assert isinstance(expert_tokens, list)

    def test_quant_optional_inputs_generated(self):
        definition = _definition()
        workload = _find_workload(
            lambda w: "scale" in w.inputs and "deq_scale1" in w.inputs
        )
        result = gen_inputs(definition, workload, "cpu", 42)
        input_names = list(definition.inputs.keys())
        for name in ("scale", "offset", "deq_scale1", "deq_scale2"):
            idx = input_names.index(name)
            tensor = result[idx]
            assert tensor is not None
            assert isinstance(tensor, torch.Tensor)
            expected_dtype = dtype_str_to_torch_dtype(workload.inputs[name].dtype)
            assert tensor.dtype == expected_dtype
            assert tensor.shape == tuple(workload.inputs[name].shape)

    def test_int8_weights_are_small_integers(self):
        definition = _definition()
        workload = _find_workload(lambda w: w.inputs["x"].dtype == "int8")
        result = gen_inputs(definition, workload, "cpu", 42)
        input_names = list(definition.inputs.keys())
        for name in ("x", "weight1", "weight2"):
            tensor = result[input_names.index(name)]
            assert tensor.dtype == torch.int8
            assert tensor.min().item() >= -5
            assert tensor.max().item() <= 4

    def test_int_bias_is_non_negative_small(self):
        definition = _definition()
        workload = _find_workload(
            lambda w: "bias1" in w.inputs and w.inputs["bias1"].dtype == "int32"
        )
        result = gen_inputs(definition, workload, "cpu", 42)
        input_names = list(definition.inputs.keys())
        for name in ("bias1", "bias2"):
            tensor = result[input_names.index(name)]
            assert tensor.dtype == torch.int32
            assert tensor.min().item() >= 0
            assert tensor.max().item() <= 4

    def test_reproducible(self):
        definition = _definition()
        workload = _find_workload(lambda w: "bias1" in w.inputs)
        r1 = gen_inputs(definition, workload, "cpu", 123)
        r2 = gen_inputs(definition, workload, "cpu", 123)
        for a, b in zip(r1, r2):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)
            else:
                assert a == b
