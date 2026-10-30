"""Tests for the MoeComputeExpertTokens custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.moe_compute_expert_tokens import (
    MoeComputeExpertTokensInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/MoeComputeExpertTokens"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestMoeComputeExpertTokensInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, MoeComputeExpertTokensInputGenerator)

    def test_sorted_experts_are_non_decreasing(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        sorted_experts = result[0]
        num_expert = result[1]
        assert torch.all(sorted_experts[1:] >= sorted_experts[:-1])
        assert sorted_experts.min().item() >= 0
        assert sorted_experts.max().item() < num_expert

    def test_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[0].shape == tuple(workload.inputs["sorted_expert_for_source_row"].shape)
        assert result[0].dtype == torch.int32
        assert result[1] == workload.inputs["num_expert"].value

    def test_reproducible(self):
        definition = _definition()
        workload = _first_workload()
        r1 = gen_inputs(definition, workload, "cpu", 123)
        r2 = gen_inputs(definition, workload, "cpu", 123)
        assert torch.equal(r1[0], r2[0])
        assert r1[1] == r2[1]
