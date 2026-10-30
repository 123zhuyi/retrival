"""Tests for the TanhGatedResidualAddBackward custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.tanh_gated_residual_add_backward import (
    TanhGatedResidualAddBackwardInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/TanhGatedResidualAddBackward"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestTanhGatedResidualAddBackwardInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, TanhGatedResidualAddBackwardInputGenerator)

    def test_mask_is_binary_bfloat16(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        mask = result[names.index("mask")]
        assert mask.dtype == torch.bfloat16
        assert mask.shape == tuple(workload.inputs["mask"].shape)
        assert set(mask.view(-1).tolist()).issubset({0.0, 1.0})

    def test_other_tensors_are_bfloat16(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for tensor_name in ("grad_output", "gate", "hidden_states"):
            tensor = result[names.index(tensor_name)]
            assert tensor.dtype == torch.bfloat16
            assert tensor.shape == tuple(workload.inputs[tensor_name].shape)

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
