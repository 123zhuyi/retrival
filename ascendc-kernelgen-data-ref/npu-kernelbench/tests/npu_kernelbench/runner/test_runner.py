"""Tests for Runner abstractions and parallelism helpers."""

import multiprocessing as mp

import pytest

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import Workload
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.data.trace import Trace, Evaluation, EvaluationStatus
from npu_kernelbench.runner.runner import Runner
from npu_kernelbench.runner.runner import _partition


def _make_dummy_workload(uuid: str) -> Workload:
    return Workload(uuid=uuid, inputs={})


def _make_dummy_definition() -> Definition:
    return Definition(
        name="test_def",
        axes={},
        inputs={},
        outputs={"out": {"shape": [], "dtype": "float32"}},
        reference="def run(): return None",
    )


def _make_dummy_solution() -> Solution:
    return Solution(
        name="test_sol",
        definition="test_def",
        author="test",
        spec={
            "languages": ["python"],
            "entry_point": "test.py::run",
            "target_hardware": ["LOCAL"],
        },
        sources=[{"path": "test.py", "content": "def run(): pass"}],
    )


class TestPartition:
    def test_even_distribution(self):
        """Every bucket gets floor or ceil(N/M) workloads."""
        wls = [_make_dummy_workload(f"w{i}") for i in range(8)]
        buckets = _partition(wls, 3)
        assert len(buckets) == 3
        assert [len(b) for b in buckets] == [3, 3, 2]

    def test_single_bucket(self):
        wls = [_make_dummy_workload(f"w{i}") for i in range(5)]
        buckets = _partition(wls, 1)
        assert len(buckets) == 1
        assert len(buckets[0]) == 5

    def test_empty_input(self):
        buckets = _partition([], 3)
        assert buckets == [[], [], []]

    def test_round_robin_order(self):
        wls = [_make_dummy_workload(f"w{i}") for i in range(7)]
        buckets = _partition(wls, 3)
        uuids = [[wl.uuid for wl in b] for b in buckets]
        assert uuids[0] == ["w0", "w3", "w6"]
        assert uuids[1] == ["w1", "w4"]
        assert uuids[2] == ["w2", "w5"]

    def test_more_buckets_than_workloads(self):
        wls = [_make_dummy_workload(f"w{i}") for i in range(2)]
        buckets = _partition(wls, 5)
        non_empty = [b for b in buckets if b]
        assert len(non_empty) == 2


class TestRunnerMakeErrorTrace:
    def test_produces_valid_trace(self):
        definition = _make_dummy_definition()
        workload = _make_dummy_workload("test-wl")
        solution = _make_dummy_solution()

        trace = Runner._make_error_trace(
            definition, workload, solution, "TIMEOUT", "test error",
        )
        assert isinstance(trace, Trace)
        assert trace.evaluation is not None
        assert trace.evaluation.status == EvaluationStatus.TIMEOUT
        assert trace.evaluation.log == "test error"
        assert trace.definition == "test_def"
        assert trace.solution == "test_sol"

    def test_unknown_status_degrades_to_runtime_error(self):
        definition = _make_dummy_definition()
        workload = _make_dummy_workload("test-wl")
        solution = _make_dummy_solution()

        trace = Runner._make_error_trace(
            definition, workload, solution, "NOT_A_REAL_STATUS", "msg",
        )
        assert trace.evaluation is not None
        assert trace.evaluation.status == EvaluationStatus.RUNTIME_ERROR
