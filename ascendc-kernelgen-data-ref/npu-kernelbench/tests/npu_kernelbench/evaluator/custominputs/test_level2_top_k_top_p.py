"""Tests for the TopKTopP custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.top_k_top_p import (
    TopKTopPInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path("/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/TopKTopP")


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


class TestTopKTopPInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, TopKTopPInputGenerator)

    def test_k_is_clamped(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        logits, p, k = result
        N = logits.shape[-1]
        assert k.dtype == torch.int32
        assert k.min().item() >= 1
        assert k.max().item() <= min(1024, N)

    def test_p_is_in_zero_one(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        p = result[1]
        assert p.min().item() >= 0.0
        assert p.max().item() <= 1.0

    def test_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[0].shape == tuple(workload.inputs["logits"].shape)
        assert result[0].dtype == torch.float16
        assert result[1].shape == tuple(workload.inputs["p"].shape)
        assert result[1].dtype == torch.float16
        assert result[2].shape == tuple(workload.inputs["k"].shape)
        assert result[2].dtype == torch.int32

    def test_reproducible(self):
        definition = _definition()
        workload = _first_workload()
        r1 = gen_inputs(definition, workload, "cpu", 123)
        r2 = gen_inputs(definition, workload, "cpu", 123)
        for a, b in zip(r1, r2):
            assert torch.equal(a, b)
