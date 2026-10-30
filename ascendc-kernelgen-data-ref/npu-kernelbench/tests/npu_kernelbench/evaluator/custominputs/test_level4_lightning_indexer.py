"""Tests for the LightningIndexer custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.lightning_indexer import (
    LightningIndexerInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "LightningIndexer"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestLightningIndexerInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, LightningIndexerInputGenerator)

    def test_query_key_weights_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for tensor_name in ("query", "key", "weights"):
            tensor = result[names.index(tensor_name)]
            assert tuple(tensor.shape) == tuple(workload.inputs[tensor_name].shape)
            assert tensor.dtype == torch.bfloat16

    def test_actual_seq_lengths_are_int32_tensors(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        q_len = result[names.index("actual_seq_lengths_query")]
        k_len = result[names.index("actual_seq_lengths_key")]
        assert isinstance(q_len, torch.Tensor)
        assert isinstance(k_len, torch.Tensor)
        assert q_len.dtype == torch.int32
        assert k_len.dtype == torch.int32

    def test_workload_with_block_table_generates_without_error(self):
        definition = _definition()
        # Find a workload that includes a block table
        with open(DATA_DIR / "workload.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                workload = Workload(**json.loads(line))
                if "block_table" in workload.inputs:
                    break
        result = gen_inputs(definition, workload, "cpu", 42)
        assert len(result) == len(definition.inputs)

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
