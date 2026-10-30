"""Tests for the MoeFinalizeRouting custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import RandomInput, Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.moe_finalize_routing import (
    MoeFinalizeRoutingInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/MoeFinalizeRouting"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


def _workload_with_optional_none():
    """Workload where skip1/skip2/bias shapes are None."""
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    wl = Workload(**json.loads(line))
    for name in ("skip1", "skip2", "bias"):
        wl.inputs[name] = RandomInput(shape=None, dtype="float16")
    return wl


class TestMoeFinalizeRoutingInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, MoeFinalizeRoutingInputGenerator)

    def test_expanded_src_to_dst_row_range(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        expanded_permuted_rows = result[names.index("expanded_permuted_rows")]
        idx = result[names.index("expanded_src_to_dst_row")]
        upper = expanded_permuted_rows.shape[0]
        assert idx.dtype == torch.int32
        assert idx.min().item() >= 0
        assert idx.max().item() < upper

    def test_optional_tensors_none_when_shape_missing(self):
        definition = _definition()
        workload = _workload_with_optional_none()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for name in ("skip1", "skip2", "bias"):
            assert result[names.index(name)] is None

    def test_export_for_source_row_range(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        export = result[names.index("export_for_source_row")]
        assert export.dtype == torch.int32
        assert export.min().item() >= 0
        assert export.max().item() < 8

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
