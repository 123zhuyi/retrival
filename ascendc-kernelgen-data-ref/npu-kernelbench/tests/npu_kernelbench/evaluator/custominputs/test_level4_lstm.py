"""Tests for the LSTM custom input generator."""

import json
from pathlib import Path

import pytest
import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import RandomInput, ScalarInput, Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level4.lstm import LSTMInputGenerator
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(__file__).resolve().parents[5] / "npu-kernelbench" / "data" / "kernel_generator" / "level4" / "LSTM"


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


def _first_workload_without_bias():
    with open(DATA_DIR / "workload.jsonl") as f:
        for line in f:
            if line.strip():
                wl = Workload(**json.loads(line))
                if isinstance(wl.inputs["bias_ih_l0"], ScalarInput) and wl.inputs["bias_ih_l0"].value is None:
                    return wl
    raise RuntimeError("No workload without bias found")


class TestLSTMInputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, LSTMInputGenerator)

    def test_bias_none_when_scalar_null(self):
        definition = _definition()
        workload = _first_workload_without_bias()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        assert result[names.index("bias_ih_l0")] is None
        assert result[names.index("bias_hh_l0")] is None

    def test_bias_present_when_random(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        bias_ih = result[names.index("bias_ih_l0")]
        bias_hh = result[names.index("bias_hh_l0")]
        assert isinstance(bias_ih, torch.Tensor)
        assert isinstance(bias_hh, torch.Tensor)
        assert bias_ih.dtype == torch.float16
        assert bias_hh.dtype == torch.float16

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
