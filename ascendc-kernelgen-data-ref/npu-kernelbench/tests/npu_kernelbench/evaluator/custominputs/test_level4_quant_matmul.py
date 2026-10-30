"""Tests for the QuantMatmul custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.quant_matmul import (
    QuantMatmulInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "QuantMatmul"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestQuantMatmulInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, QuantMatmulInputGenerator)

    def test_x1_x2_are_int8(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        x1 = result[names.index("x1")]
        x2 = result[names.index("x2")]
        assert x1.dtype == torch.int8
        assert x2.dtype == torch.int8
        assert tuple(x1.shape) == tuple(workload.inputs["x1"].shape)
        assert tuple(x2.shape) == tuple(workload.inputs["x2"].shape)

    def test_scale_is_float(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        scale = result[names.index("scale")]
        assert scale.dtype == torch.bfloat16
        assert tuple(scale.shape) == tuple(workload.inputs["scale"].shape)

    def test_output_dtype_scalar_passed_through(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        output_dtype = result[names.index("output_dtype")]
        assert output_dtype == torch.bfloat16

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
