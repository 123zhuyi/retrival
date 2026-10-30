"""Tests for score computation."""

from npu_kernelbench.report.score import (
    ScoreSummary,
    compute_score,
    format_score_summary,
)
from npu_kernelbench.data.trace import (
    Trace, Evaluation, EvaluationStatus, Correctness, Performance, Environment,
)
from npu_kernelbench.data.workload import Workload, RandomInput


ENV = Environment(hardware="Ascend_910B", libs={})


def _make_trace(status=EvaluationStatus.PASSED, latency=1.0, speedup=2.0):
    wl = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="test")
    perf = Performance(latency_ms=latency, speedup_factor=speedup)
    corr = Correctness()
    return Trace(
        definition="test",
        workload=wl,
        solution="sol",
        evaluation=Evaluation(
            status=status,
            environment=ENV,
            timestamp="2026-01-01T00:00:00Z",
            correctness=corr if status in (EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL) else None,
            performance=perf if status == EvaluationStatus.PASSED else None,
        ),
    )


class TestScoreSummary:
    def test_empty(self):
        s = ScoreSummary()
        assert s.total == 0
        assert s.pass_rate == 0.0


class TestComputeScore:
    def test_all_passed(self):
        traces = [_make_trace() for _ in range(5)]
        s = compute_score(traces)
        assert s.total == 5
        assert s.passed == 5
        assert s.pass_rate == 1.0
        assert s.avg_speedup == 2.0

    def test_mixed(self):
        traces = [
            _make_trace(EvaluationStatus.PASSED),
            _make_trace(EvaluationStatus.PASSED),
            _make_trace(EvaluationStatus.RUNTIME_ERROR),
            _make_trace(EvaluationStatus.INCORRECT_NUMERICAL),
        ]
        s = compute_score(traces)
        assert s.total == 4
        assert s.passed == 2
        assert s.runtime_errors == 1
        assert s.incorrect_numerical == 1
        assert s.pass_rate == 0.5

    def test_no_speedup_when_zero(self):
        traces = [_make_trace(speedup=0.0)]
        s = compute_score(traces)
        assert s.avg_speedup is None

    def test_all_status_types(self):
        traces = [
            _make_trace(EvaluationStatus.PASSED),
            _make_trace(EvaluationStatus.RUNTIME_ERROR),
            _make_trace(EvaluationStatus.COMPILE_ERROR),
            _make_trace(EvaluationStatus.INCORRECT_NUMERICAL),
            _make_trace(EvaluationStatus.INCORRECT_SHAPE),
            _make_trace(EvaluationStatus.INCORRECT_DTYPE),
            _make_trace(EvaluationStatus.INVALID_REFERENCE),
            _make_trace(EvaluationStatus.TIMEOUT),
            _make_trace(EvaluationStatus.REWARD_HACK),
            _make_trace(EvaluationStatus.CRASH),
        ]
        s = compute_score(traces)
        assert s.passed == 1
        assert s.runtime_errors == 1
        assert s.compile_errors == 1
        assert s.incorrect_numerical == 1
        assert s.incorrect_shape == 1
        assert s.incorrect_dtype == 1
        assert s.invalid_reference == 1
        assert s.timeout == 1
        assert s.reward_hacks == 1
        assert s.crashes == 1
        assert s.pass_rate == 1 / 10


class TestFormatScoreSummary:
    def test_basic(self):
        s = ScoreSummary(total=10, passed=8, pass_rate=0.8, avg_speedup=1.5)
        output = format_score_summary(s)
        assert "8/10" not in output  # old format
        assert "80.0%" in output
        assert "1.50x" in output
