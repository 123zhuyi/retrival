"""Tests for resume support (inlined in cli/main.py)."""

import logging
from pathlib import Path

from npu_kernelbench.cli.main import (
    _filter_workloads as filter_workloads,
    _load_resume_index as load_resume_index,
    _make_resume_key as make_key,
)
from npu_kernelbench.data import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from npu_kernelbench.data.workload import RandomInput, Workload

ENV = Environment(hardware="Ascend_910B", libs={"torch": "2.9.0"})


def _wl(uuid: str) -> Workload:
    return Workload(
        axes={"N": 8}, inputs={"x": RandomInput(dtype="float32")}, uuid=uuid
    )


def _passed_trace(definition: str, solution: str, uuid: str) -> Trace:
    return Trace(
        definition=definition,
        workload=_wl(uuid),
        solution=solution,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=ENV,
            timestamp="2026-06-14T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(latency_ms=0.1),
        ),
    )


def _error_trace(definition: str, solution: str, uuid: str) -> Trace:
    return Trace(
        definition=definition,
        workload=_wl(uuid),
        solution=solution,
        evaluation=Evaluation(
            status=EvaluationStatus.RUNTIME_ERROR,
            environment=ENV,
            timestamp="2026-06-14T00:00:00Z",
            log="boom",
        ),
    )


def _workload_only_trace(definition: str, uuid: str) -> Trace:
    return Trace(definition=definition, workload=_wl(uuid))


class TestMakeKey:
    def test_key_is_three_tuple(self):
        assert make_key("gelu", "gelu_v1", "wl-1") == ("gelu", "gelu_v1", "wl-1")


class TestLoadResumeIndex:
    def test_missing_file_returns_empty(self):
        assert load_resume_index(Path("/nonexistent/file.jsonl")) == {}

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text("")
        assert load_resume_index(p) == {}

    def test_indexes_passed_traces(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(_passed_trace("gelu", "gelu_v1", "wl-1").model_dump_json() + "\n")
        index = load_resume_index(p)
        assert make_key("gelu", "gelu_v1", "wl-1") in index

    def test_indexes_error_traces_too(self, tmp_path):
        # skip-all-existing: an errored result still counts as done
        p = tmp_path / "out.jsonl"
        p.write_text(_error_trace("gelu", "gelu_v1", "wl-1").model_dump_json() + "\n")
        assert make_key("gelu", "gelu_v1", "wl-1") in load_resume_index(p)

    def test_workload_only_trace_not_indexed(self, tmp_path):
        p = tmp_path / "out.jsonl"
        p.write_text(_workload_only_trace("gelu", "wl-1").model_dump_json() + "\n")
        assert load_resume_index(p) == {}

    def test_corrupt_trailing_line_skipped_with_warning(self, tmp_path, caplog):
        p = tmp_path / "out.jsonl"
        good = _passed_trace("gelu", "gelu_v1", "wl-1")
        # valid line, then a half-written line (killed mid-write)
        p.write_text(
            good.model_dump_json() + "\n"
            + '{"definition":"gelu","workload":{"uuid":"wl-2"'
        )
        with caplog.at_level(logging.WARNING):
            index = load_resume_index(p)
        assert make_key("gelu", "gelu_v1", "wl-1") in index
        assert make_key("gelu", "gelu_v1", "wl-2") not in index
        assert any(
            "skipping unparseable" in r.getMessage() for r in caplog.records
        )


class TestFilterWorkloads:
    def test_all_pending_when_index_empty(self):
        wls = [_wl("w1"), _wl("w2")]
        pending, reused = filter_workloads(wls, "gelu", "gelu_v1", {})
        assert [w.uuid for w in pending] == ["w1", "w2"]
        assert reused == []

    def test_partial_done(self):
        wls = [_wl("w1"), _wl("w2"), _wl("w3")]
        index = {
            make_key("gelu", "gelu_v1", "w1"): _passed_trace("gelu", "gelu_v1", "w1"),
        }
        pending, reused = filter_workloads(wls, "gelu", "gelu_v1", index)
        assert [w.uuid for w in pending] == ["w2", "w3"]
        assert [t.workload.uuid for t in reused] == ["w1"]

    def test_all_done_pending_empty(self):
        wls = [_wl("w1"), _wl("w2")]
        index = {
            make_key("gelu", "gelu_v1", "w1"): _passed_trace("gelu", "gelu_v1", "w1"),
            make_key("gelu", "gelu_v1", "w2"): _passed_trace("gelu", "gelu_v1", "w2"),
        }
        pending, reused = filter_workloads(wls, "gelu", "gelu_v1", index)
        assert pending == []
        assert [t.workload.uuid for t in reused] == ["w1", "w2"]

    def test_reused_preserves_workload_order(self):
        wls = [_wl("w1"), _wl("w2"), _wl("w3")]
        index = {
            make_key("gelu", "gelu_v1", "w3"): _passed_trace("gelu", "gelu_v1", "w3"),
            make_key("gelu", "gelu_v1", "w1"): _passed_trace("gelu", "gelu_v1", "w1"),
        }
        _, reused = filter_workloads(wls, "gelu", "gelu_v1", index)
        assert [t.workload.uuid for t in reused] == ["w1", "w3"]

    def test_different_solution_same_uuid_not_matched(self):
        # uuid w1 done for sol_A must NOT skip w1 for sol_B
        wls = [_wl("w1")]
        index = {make_key("gelu", "sol_A", "w1"): _passed_trace("gelu", "sol_A", "w1")}
        pending, reused = filter_workloads(wls, "gelu", "sol_B", index)
        assert [w.uuid for w in pending] == ["w1"]
        assert reused == []
