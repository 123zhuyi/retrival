"""Tests for Trace, Correctness, Performance, and Evaluation models."""

import pytest
from pydantic import ValidationError

from npu_kernelbench.data.trace import (
    Correctness,
    Performance,
    Environment,
    Evaluation,
    EvaluationStatus,
    Trace,
)
from npu_kernelbench.data.workload import Workload, RandomInput


ENV = Environment(hardware="Ascend_910B", libs={"torch": "2.9.0", "torch_npu": "2.9.0"})


class TestCorrectness:
    def test_defaults(self):
        c = Correctness()
        assert c.max_absolute_error == 0.0
        assert c.max_relative_error == 0.0
        assert c.has_nan is False
        assert c.has_inf is False

    def test_negative_error_rejected(self):
        with pytest.raises(ValidationError):
            Correctness(max_absolute_error=-1.0)

    def test_error_values(self):
        c = Correctness(max_absolute_error=0.001, max_relative_error=0.01)
        assert c.max_absolute_error == 0.001


class TestPerformance:
    def test_defaults(self):
        p = Performance()
        assert p.latency_ms == 0.0

    def test_with_speedup(self):
        p = Performance(latency_ms=0.5, reference_latency_ms=1.0, speedup_factor=2.0)
        assert p.speedup_factor == 2.0


class TestEvaluation:
    def test_passed_evaluation(self):
        ev = Evaluation(
            status=EvaluationStatus.PASSED,
            environment=ENV,
            timestamp="2026-06-03T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(),
        )
        assert ev.status == EvaluationStatus.PASSED

    def test_passed_without_correctness_rejected(self):
        with pytest.raises(ValidationError):
            Evaluation(
                status=EvaluationStatus.PASSED,
                environment=ENV,
                timestamp="2026-06-03T00:00:00Z",
                performance=Performance(),
            )

    def test_passed_without_performance_rejected(self):
        with pytest.raises(ValidationError):
            Evaluation(
                status=EvaluationStatus.PASSED,
                environment=ENV,
                timestamp="2026-06-03T00:00:00Z",
                correctness=Correctness(),
            )

    def test_runtime_error_no_correctness(self):
        ev = Evaluation(
            status=EvaluationStatus.RUNTIME_ERROR,
            environment=ENV,
            timestamp="2026-06-03T00:00:00Z",
            log="Segmentation fault",
        )
        assert ev.correctness is None
        assert ev.performance is None

    def test_incorrect_numerical_has_correctness_no_performance(self):
        ev = Evaluation(
            status=EvaluationStatus.INCORRECT_NUMERICAL,
            environment=ENV,
            timestamp="2026-06-03T00:00:00Z",
            correctness=Correctness(max_absolute_error=100.0),
        )
        assert ev.correctness is not None
        assert ev.performance is None


class TestTrace:
    def test_full_trace(self):
        wl = Workload(
            axes={"N": 32}, inputs={"x": RandomInput()}, uuid="wl-1"
        )
        trace = Trace(
            definition="gelu",
            workload=wl,
            solution="gelu_v1",
            evaluation=Evaluation(
                status=EvaluationStatus.PASSED,
                environment=ENV,
                timestamp="2026-06-03T00:00:00Z",
                correctness=Correctness(),
                performance=Performance(latency_ms=0.1),
            ),
        )
        assert trace.is_workload_trace() is False
        assert trace.is_successful() is True

    def test_workload_trace(self):
        wl = Workload(
            axes={"N": 32}, inputs={"x": RandomInput()}, uuid="wl-1"
        )
        trace = Trace(definition="gelu", workload=wl)
        assert trace.is_workload_trace() is True
        assert trace.is_successful() is False


class TestCrashErrorTrace:
    """Regression: the worker-crash error path must produce a valid Trace.

    ``Runner._make_error_trace`` is invoked when a worker process hard-crashes
    (e.g. a C++ segfault). Because it *is* the error path, it must never raise —
    otherwise the secondary exception masks the real worker failure. Previously
    it called ``EvaluationStatus("WORKER_CRASH")`` with no such member, so the
    error path itself crashed.
    """

    def test_crash_is_a_valid_status(self):
        assert EvaluationStatus("CRASH") is EvaluationStatus.CRASH

    def test_make_error_trace_crash_status(self):
        from types import SimpleNamespace

        from npu_kernelbench.runner.runner import Runner

        wl = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="crash-wl")
        definition = SimpleNamespace(name="test_def")
        solution = SimpleNamespace(name="test_sol")

        trace = Runner._make_error_trace(definition, wl, solution, "CRASH", "boom")
        assert trace.evaluation.status is EvaluationStatus.CRASH
        assert trace.definition == "test_def"
        assert trace.solution == "test_sol"
        assert trace.evaluation.log == "boom"

    def test_make_error_trace_unknown_status_does_not_raise(self):
        from types import SimpleNamespace

        from npu_kernelbench.runner.runner import Runner

        wl = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="crash-wl")
        definition = SimpleNamespace(name="test_def")
        solution = SimpleNamespace(name="test_sol")

        # A typo'd / unknown status must degrade to RUNTIME_ERROR, never raise.
        trace = Runner._make_error_trace(
            definition, wl, solution, "WORKER_CRASH_TYPO", "boom"
        )
        assert trace.evaluation.status is EvaluationStatus.RUNTIME_ERROR
