"""Tests for the MultiMaskAttentionAggregation custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.multi_mask_attention_aggregation import (
    MultiMaskAttentionAggregationInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/MultiMaskAttentionAggregation"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestMultiMaskAttentionAggregationInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, MultiMaskAttentionAggregationInputGenerator)

    def test_ref_target_masks_are_binary_float32(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        masks = result[names.index("ref_target_masks")]
        assert masks.dtype == torch.float32
        assert masks.shape == tuple(workload.inputs["ref_target_masks"].shape)
        assert set(masks.flatten().tolist()).issubset({0.0, 1.0})

    def test_attn_dtype_shape_unchanged(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        attn = result[names.index("attn")]
        assert attn.dtype == torch.float32
        assert attn.shape == tuple(workload.inputs["attn"].shape)

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
