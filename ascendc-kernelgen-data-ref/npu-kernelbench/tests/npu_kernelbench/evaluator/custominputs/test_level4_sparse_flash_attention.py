"""Tests for the SparseFlashAttention custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.sparse_flash_attention import (
    SparseFlashAttentionInputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "SparseFlashAttention"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


class TestSparseFlashAttentionInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, SparseFlashAttentionInputGenerator)

    def test_query_key_value_shapes_and_dtypes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        for tensor_name in ("query", "key", "value"):
            tensor = result[names.index(tensor_name)]
            assert tuple(tensor.shape) == tuple(workload.inputs[tensor_name].shape)
            assert tensor.dtype == torch.bfloat16

    def test_sparse_indices_int32(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        sparse_indices = result[names.index("sparse_indices")]
        assert sparse_indices.dtype == torch.int32
        assert tuple(sparse_indices.shape) == tuple(workload.inputs["sparse_indices"].shape)

    def test_actual_seq_lengths_are_int32_tensors(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        q_len = result[names.index("actual_seq_lengths_query")]
        kv_len = result[names.index("actual_seq_lengths_kv")]
        assert isinstance(q_len, torch.Tensor)
        assert isinstance(kv_len, torch.Tensor)
        assert q_len.dtype == torch.int32
        assert kv_len.dtype == torch.int32

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
