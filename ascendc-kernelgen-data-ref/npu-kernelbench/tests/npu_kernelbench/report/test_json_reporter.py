"""Tests for JsonReporter."""

import json

from npu_kernelbench.report.json_reporter import JsonReporter
from npu_kernelbench.data.trace import (
    Trace, Evaluation, EvaluationStatus, Correctness, Performance, Environment,
)
from npu_kernelbench.data.workload import Workload, RandomInput


ENV = Environment(hardware="Ascend_910B")
WL = Workload(axes={"N": 32}, inputs={"x": RandomInput()}, uuid="test-wl")


def _make_trace(status=EvaluationStatus.PASSED, latency=1.0, speedup=2.0):
    perf = Performance(latency_ms=latency, speedup_factor=speedup)
    is_pass = status == EvaluationStatus.PASSED
    return Trace(
        definition="test",
        workload=WL,
        solution="sol",
        evaluation=Evaluation(
            status=status,
            environment=ENV,
            timestamp="2026-01-01T00:00:00Z",
            correctness=Correctness() if is_pass else None,
            performance=perf if is_pass else None,
        ),
    )


class TestJsonReporter:
    def test_write_report_creates_file(self, tmp_path):
        traces = [_make_trace() for _ in range(3)]
        path = tmp_path / "report.json"
        JsonReporter().write_report(traces, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["traces"]) == 3
        assert "score_summary" in data

    def test_write_report_without_score(self, tmp_path):
        traces = [_make_trace()]
        path = tmp_path / "report.json"
        JsonReporter().write_report(traces, path, include_score=False)
        data = json.loads(path.read_text())
        assert "traces" in data
        assert "score_summary" not in data

    def test_append_trace(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        reporter = JsonReporter()
        reporter.append_trace(_make_trace(), path)
        reporter.append_trace(_make_trace(EvaluationStatus.RUNTIME_ERROR), path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert obj["definition"] == "test"
