"""Tests for the CTCLoss custom input generator (migrated from CTClossV3)."""

import json
from pathlib import Path

import torch

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.evaluator.custominputs import get_input_generator_registry
from npu_kernelbench.evaluator.custominputs.level1.ctcloss_v3 import (
    CTClossV3InputGenerator,
)
from npu_kernelbench.evaluator.generator import gen_inputs


DATA_DIR = Path(
    "/mnt/l00956942/ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level1/CTCLoss"
)


def _definition():
    return Definition(**json.loads((DATA_DIR / "definition.json").read_text()))


def _first_workload():
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    return Workload(**json.loads(line))


def _make_2d_workload():
    """Create a synthetic 2D workload for branch coverage."""
    with open(DATA_DIR / "workload.jsonl") as f:
        line = next(l for l in f if l.strip())
    wl = Workload(**json.loads(line))
    wl.inputs["log_probs"].shape = [10, 5]
    wl.inputs["targets"].shape = [3]
    return wl


class TestCTClossV3InputGenerator:
    def test_registry_resolves_to_custom_generator(self):
        registry = get_input_generator_registry()
        generator = registry.resolve(_definition())
        assert isinstance(generator, CTClossV3InputGenerator)

    def test_log_probs_are_log_softmax(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        log_probs = result[names.index("log_probs")]
        assert torch.allclose(
            torch.exp(log_probs).sum(dim=-1),
            torch.ones_like(torch.exp(log_probs).sum(dim=-1)),
            atol=1e-5,
        )

    def test_targets_exclude_blank(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        targets = result[names.index("targets")]
        blank = workload.inputs["blank"].value
        C = workload.inputs["log_probs"].shape[-1]
        assert targets.dtype == torch.int64
        assert targets.min().item() >= 0
        assert targets.max().item() < C
        assert blank not in targets.tolist()

    def test_3d_shapes(self):
        definition = _definition()
        workload = _first_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        log_probs = result[names.index("log_probs")]
        targets = result[names.index("targets")]
        assert log_probs.dim() == 3
        assert targets.dim() == 2

    def test_2d_shapes(self):
        definition = _definition()
        workload = _make_2d_workload()
        result = gen_inputs(definition, workload, "cpu", 42)
        names = list(definition.inputs)
        log_probs = result[names.index("log_probs")]
        targets = result[names.index("targets")]
        assert log_probs.dim() == 2
        assert targets.dim() == 1

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
