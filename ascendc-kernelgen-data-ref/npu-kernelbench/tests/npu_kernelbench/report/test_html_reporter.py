"""Tests for HtmlReporter."""

from npu_kernelbench.report.html_reporter import HtmlReporter
from npu_kernelbench.data.trace import (
    Trace,
    Evaluation,
    EvaluationStatus,
    Correctness,
    Performance,
    Environment,
)
from npu_kernelbench.data.workload import Workload, RandomInput


ENV = Environment(hardware="Ascend_910B")
WL = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="test-wl")


def _make_trace(
    solution: str = "sol",
    status: EvaluationStatus = EvaluationStatus.PASSED,
    latency: float | None = 1.0,
    ref_latency: float | None = 2.0,
    speedup: float | None = 2.0,
    abs_err: float = 1e-6,
    rel_err: float = 1e-6,
) -> Trace:
    """Build a Trace with optional correctness/performance per status rules."""
    correctness = None
    performance = None

    if status in (EvaluationStatus.PASSED, EvaluationStatus.INCORRECT_NUMERICAL):
        correctness = Correctness(
            max_absolute_error=abs_err,
            max_relative_error=rel_err,
        )

    if status == EvaluationStatus.PASSED and latency is not None and ref_latency is not None and speedup is not None:
        performance = Performance(
            latency_ms=latency,
            reference_latency_ms=ref_latency,
            speedup_factor=speedup,
        )

    return Trace(
        definition="test",
        workload=WL,
        solution=solution,
        evaluation=Evaluation(
            status=status,
            environment=ENV,
            timestamp="2026-01-01T00:00:00Z",
            correctness=correctness,
            performance=performance,
            log="some output",
        ),
    )


class TestHtmlReporter:
    """Integration tests for write_report output."""

    def test_write_report_creates_html_file(self, tmp_path):
        traces = [_make_trace() for _ in range(3)]
        path = tmp_path / "report.html"
        HtmlReporter().write_report(traces, path)
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "NPU KernelBench Report" in content
        assert "PASSED" in content

    def test_report_with_errors(self, tmp_path):
        traces = [
            _make_trace(status=EvaluationStatus.PASSED),
            _make_trace(status=EvaluationStatus.RUNTIME_ERROR, latency=None),
        ]
        path = tmp_path / "report.html"
        HtmlReporter().write_report(traces, path)
        content = path.read_text()
        assert "RUNTIME_ERROR" in content

    def test_custom_title(self, tmp_path):
        traces = [_make_trace()]
        path = tmp_path / "report.html"
        HtmlReporter().write_report(traces, path, title="Custom Title")
        content = path.read_text()
        assert "<title>Custom Title</title>" in content


class TestExtractRows:
    """Unit tests for _extract_rows() returning list of dicts."""

    def test_single_passed_trace(self):
        traces = [_make_trace()]
        rows = HtmlReporter._extract_rows(traces)
        assert len(rows) == 1
        assert rows[0]["status"] == "PASSED"
        assert rows[0]["latency"] == "1.000"
        assert rows[0]["ref_latency"] == "2.000"
        assert rows[0]["speedup"] == "2.00x"
        assert rows[0]["abs_err"] == "1.00e-06"
        assert rows[0]["rel_err"] == "1.00e-06"
        assert rows[0]["log"] == "some output"

    def test_no_evaluation(self):
        trace = Trace(
            definition="test",
            workload=WL,
            solution="s1",
            evaluation=None,
        )
        rows = HtmlReporter._extract_rows([trace])
        assert rows[0]["status"] == ""
        assert rows[0]["latency"] == ""

    def test_nan_correctness(self):
        trace = _make_trace(status=EvaluationStatus.INCORRECT_NUMERICAL)
        trace.evaluation.correctness.has_nan = True
        rows = HtmlReporter._extract_rows([trace])
        assert rows[0]["abs_err"] == "NaN"
        assert rows[0]["rel_err"] == "NaN"

    def test_inf_correctness(self):
        trace = _make_trace(status=EvaluationStatus.INCORRECT_NUMERICAL)
        trace.evaluation.correctness.has_inf = True
        trace.evaluation.correctness.has_nan = False
        rows = HtmlReporter._extract_rows([trace])
        assert rows[0]["abs_err"] == "Inf"
        assert rows[0]["rel_err"] == "Inf"

    def test_log_truncated_to_120(self):
        long_log = "x" * 200
        trace = _make_trace()
        trace.evaluation.log = long_log
        rows = HtmlReporter._extract_rows([trace])
        assert rows[0]["log"] == long_log[:120]


class TestGroupTraces:
    """Unit tests for _group_traces() grouping and scoring."""

    def test_single_solution(self):
        traces = [
            _make_trace(solution="s1", status=EvaluationStatus.PASSED),
            _make_trace(solution="s1", status=EvaluationStatus.RUNTIME_ERROR),
        ]
        groups = HtmlReporter._group_traces(traces)
        assert len(groups) == 1
        g = groups[0]
        assert g["name"] == "s1"
        assert g["score"].total == 2
        assert g["score"].passed == 1
        assert g["score"].runtime_errors == 1
        assert len(g["rows"]) == 2

    def test_multiple_solutions_ordered(self):
        traces = [
            _make_trace(solution="s2"),
            _make_trace(solution="s1"),
            _make_trace(solution="s2"),
        ]
        groups = HtmlReporter._group_traces(traces)
        assert len(groups) == 2
        assert groups[0]["name"] == "s2"
        assert groups[0]["score"].total == 2
        assert groups[1]["name"] == "s1"
        assert groups[1]["score"].total == 1

    def test_null_solution_fallback(self):
        trace = Trace(
            definition="test",
            workload=WL,
            solution=None,
            evaluation=_make_trace().evaluation,
        )
        groups = HtmlReporter._group_traces([trace])
        assert groups[0]["name"] == "(no solution)"

    def test_avg_speedup_computed(self):
        traces = [
            _make_trace(solution="s1", status=EvaluationStatus.PASSED, speedup=2.0),
            _make_trace(solution="s1", status=EvaluationStatus.PASSED, speedup=3.0),
        ]
        groups = HtmlReporter._group_traces(traces)
        assert groups[0]["score"].avg_speedup == 2.5
