"""Tests for the ConvStandard3d custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level3.conv_standard3d import (
    ConvStandard3dInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "kernel_generator"
    / "level3"
    / "ConvStandard3d"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestConvStandard3dInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, ConvStandard3dInputGenerator)

    def test_x_shape_and_dtype(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[0].shape == tuple(workload.inputs["x"].shape)
        assert result[0].dtype == torch.float32
        assert result[0].ndim == 5

    def test_scalar_hyperparameters_passed_through(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[1] == workload.inputs["in_channels"].value
        assert result[2] == workload.inputs["out_channels"].value
        assert result[3] == workload.inputs["kernel_size"].value
        assert result[-1] == workload.inputs["bias"].value

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
