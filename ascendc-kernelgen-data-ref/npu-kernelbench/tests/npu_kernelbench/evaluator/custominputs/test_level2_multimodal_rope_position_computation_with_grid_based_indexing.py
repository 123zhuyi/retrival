"""Tests for the MultimodalRopePositionComputationWithGridBasedIndexing custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level2.multimodal_rope_position_computation_with_grid_based_indexing import (
    MultimodalRopePositionComputationWithGridBasedIndexingInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level2/MultimodalRopePositionComputationWithGridBasedIndexing"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestMultimodalRopePositionComputationWithGridBasedIndexingInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, MultimodalRopePositionComputationWithGridBasedIndexingInputGenerator)

    def test_grid_thw_is_multiple_of_merge_size(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        grid_thw = result[0]
        merge_size = workload.inputs["spatial_merge_size"].value
        assert grid_thw.dtype == torch.int64
        assert torch.all(grid_thw % merge_size == 0)
        assert grid_thw.min().item() >= merge_size

    def test_core_tensors_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        grid_thw, pos_embed_weight, inv_freq = result[0], result[1], result[2]
        assert grid_thw.shape == tuple(workload.inputs["grid_thw"].shape)
        assert pos_embed_weight.shape == tuple(workload.inputs["pos_embed_weight"].shape)
        assert inv_freq.shape == tuple(workload.inputs["inv_freq"].shape)

    def test_scalars_passed_through(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        assert result[3] == workload.inputs["spatial_merge_size"].value
        assert result[4] == workload.inputs["num_grid_per_side"].value
        assert result[5] == workload.inputs["mrope_section"].value

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
