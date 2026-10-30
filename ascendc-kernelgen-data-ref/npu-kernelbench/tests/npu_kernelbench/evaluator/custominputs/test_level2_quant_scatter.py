"""Tests for the QuantScatter custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.quant_scatter import (
    QuantScatterInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/QuantScatter"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestQuantScatterInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, QuantScatterInputGenerator)

    def test_input_is_small_non_negative_int8(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        input_tensor = result[0]
        assert input_tensor.dtype == torch.int8
        assert input_tensor.min().item() >= 0
        assert input_tensor.max().item() <= 127

    def test_indices_are_non_negative_and_in_range(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        indices = result[1]
        axis = workload.inputs["axis"].value
        ndim = len(workload.inputs["indices"].shape)
        if axis < 0:
            axis = ndim + axis
        max_val = workload.inputs["indices"].shape[axis]
        assert indices.dtype == torch.int32
        assert indices.min().item() >= 0
        assert indices.max().item() < max_val

    def test_float_tensors_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        updates, quant_scales, quant_zero_points = result[2], result[3], result[4]
        assert updates.shape == tuple(workload.inputs["updates"].shape)
        assert quant_scales.shape == tuple(workload.inputs["quant_scales"].shape)
        assert quant_zero_points.shape == tuple(workload.inputs["quant_zero_points"].shape)

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
