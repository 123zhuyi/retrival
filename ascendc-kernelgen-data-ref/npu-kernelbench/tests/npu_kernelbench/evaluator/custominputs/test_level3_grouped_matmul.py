"""Tests for level3 custom input generators."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level3.grouped_matmul import (
    GroupedMatmulInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "kernel_generator"
    / "level3"
    / "GroupedMatmul"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _workload(offset_type: str):
    with open(DATA_DIR / "workload.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            wl = Workload(**json.loads(line))
            spec = wl.inputs.get("offsets")
            if offset_type == "tensor" and spec.type == "random":
                return wl
            if offset_type == "none" and spec.type == "scalar" and spec.value is None:
                return wl
    raise RuntimeError(f"No workload with offsets={offset_type}")


class TestGroupedMatmulInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, GroupedMatmulInputGenerator)

    def test_2d_a_generates_even_end_offsets(self):
        definition = _definition()
        workload = _workload("tensor")
        result = gen_inputs(definition, workload, "cpu", 42)
        a, b, offsets = result
        num_groups = b.shape[0]
        group_size = a.shape[0] // num_groups
        expected = torch.tensor(
            [(i + 1) * group_size for i in range(num_groups)], dtype=torch.int32
        )
        assert a.ndim == 2
        assert offsets.dtype == torch.int32
        assert torch.equal(offsets.cpu(), expected)

    def test_3d_a_returns_none_offsets(self):
        definition = _definition()
        workload = _workload("none")
        result = gen_inputs(definition, workload, "cpu", 42)
        a, b, offsets = result
        assert a.ndim == 3
        assert offsets is None

    def test_shapes_and_dtypes(self):
        definition = _definition()
        workload = _workload("tensor")
        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[0].shape == tuple(workload.inputs["A"].shape)
        assert result[0].dtype == torch.float32
        assert result[1].shape == tuple(workload.inputs["B"].shape)
        assert result[1].dtype == torch.float32

    def test_reference_fallback_handles_end_offsets(self):
        definition = _definition()
        namespace = {}
        exec(definition.reference, namespace)

        a = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        b = torch.stack(
            [
                torch.ones(2, 1, dtype=torch.float32),
                torch.full((2, 1), 2.0, dtype=torch.float32),
            ]
        )
        offsets = torch.tensor([4, 8], dtype=torch.int32)

        result = namespace["_fallback_grouped_mm"](a, b, offsets)
        expected = torch.cat([a[0:4] @ b[0], a[4:8] @ b[1]], dim=0)
        assert torch.equal(result, expected)

    def test_reference_fallback_handles_direct_3d_grouping(self):
        definition = _definition()
        namespace = {}
        exec(definition.reference, namespace)

        a = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        b = torch.ones(2, 2, 1, dtype=torch.float32)

        result = namespace["_fallback_grouped_mm"](a, b, None)
        expected = torch.bmm(a, b)
        assert torch.equal(result, expected)
