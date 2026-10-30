"""Tests for the FusedInferAttentionScore custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.fused_infer_attention_score import (
    FusedInferAttentionScoreInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "FusedInferAttentionScore"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestFusedInferAttentionScoreInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, FusedInferAttentionScoreInputGenerator)

    def test_query_key_value_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for tensor_name in ("query", "key", "value"):
            tensor = result[names.index(tensor_name)]
            assert tuple(tensor.shape) == tuple(workload.inputs[tensor_name].shape)
            assert tensor.dtype == torch.float16

    def test_scalar_hyperparameters_passed_through(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        assert result[names.index("num_heads")] == workload.inputs["num_heads"].value
        assert result[names.index("input_layout")] == workload.inputs["input_layout"].value
        assert result[names.index("scale")] == workload.inputs["scale"].value

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
