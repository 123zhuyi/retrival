"""Integration tests for --resume in the CLI (runner mocked, no NPU)."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from npu_kernelbench.cli.main import cli
from npu_kernelbench.cli.main import _load_resume_index as load_resume_index
from npu_kernelbench.data import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
    append_jsonl_line,
)
from npu_kernelbench.data.workload import RandomInput, Workload
from npu_kernelbench.report import compute_score as _real_compute_score

ENV = Environment(hardware="Ascend_910B", libs={"torch": "2.9.0"})

DEFINITION = {
    "name": "add",
    "op_type": "add",
    "axes": {"N": {"type": "var", "description": "size"}},
    "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
    "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
    "reference": "def run(x):\n    return x\n",
}

WORKLOADS = [
    {"uuid": "w1", "axes": {"N": 8},
     "inputs": {"x": {"type": "random", "dtype": "float32"}}},
    {"uuid": "w2", "axes": {"N": 16},
     "inputs": {"x": {"type": "random", "dtype": "float32"}}},
    {"uuid": "w3", "axes": {"N": 32},
     "inputs": {"x": {"type": "random", "dtype": "float32"}}},
]

SOLUTION = {
    "name": "add_v1",
    "definition": "add",
    "author": "test",
    "spec": {
        "languages": ["python"],
        "target_hardware": ["LOCAL"],
        "entry_point": "kernel.py::run",
    },
    "sources": [{"path": "kernel.py", "content": "def run(x):\n    return x\n"}],
}


def _write_kernel(base: Path, name: str = "add") -> Path:
    """Write definition.json + workload.jsonl + solution.json under base/name."""
    kd = base / name
    kd.mkdir(parents=True)
    (kd / "definition.json").write_text(json.dumps(DEFINITION))
    (kd / "workload.jsonl").write_text(
        "\n".join(json.dumps(w) for w in WORKLOADS) + "\n"
    )
    (kd / "solution.json").write_text(json.dumps(SOLUTION))
    return kd


def _write_named_kernel(base: Path, kname: str, uuids: list[str]) -> Path:
    """Write a kernel dir base/kname with definition.name=kname + given uuids.

    Used for multi-kernel datasets where each kernel needs a distinct
    definition/solution name (so resume identity keys don't collide).
    """
    kd = base / kname
    kd.mkdir(parents=True)
    definition = {**DEFINITION, "name": kname, "op_type": kname}
    workloads = [
        {"uuid": u, "axes": {"N": 8},
         "inputs": {"x": {"type": "random", "dtype": "float32"}}}
        for u in uuids
    ]
    solution = {**SOLUTION, "name": f"{kname}_v1", "definition": kname}
    (kd / "definition.json").write_text(json.dumps(definition))
    (kd / "workload.jsonl").write_text(
        "\n".join(json.dumps(w) for w in workloads) + "\n"
    )
    (kd / "solution.json").write_text(json.dumps(solution))
    return kd


def _passed_trace(uuid: str, definition: str = "add", solution: str = "add_v1") -> Trace:
    return Trace(
        definition=definition,
        workload=Workload(
            axes={"N": 8}, inputs={"x": RandomInput(dtype="float32")}, uuid=uuid
        ),
        solution=solution,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=ENV,
            timestamp="2026-06-14T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(latency_ms=0.1),
        ),
    )


def _make_fake_runner():
    """Fake _evaluate_with_runner: records per-call uuids, appends + returns traces."""
    calls: list[list[str]] = []

    def fake(definition, workloads, solution, config, output_file=None):
        calls.append([w.uuid for w in workloads])
        traces = []
        for wl in workloads:
            t = _passed_trace(
                wl.uuid, definition=definition.name, solution=solution.name
            )
            traces.append(t)
            if output_file:
                append_jsonl_line(t, output_file)
        return traces

    return fake, calls


def _read_uuids(path: Path) -> list[str]:
    from npu_kernelbench.data import Trace as T

    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return sorted(T.model_validate_json(ln).workload.uuid for ln in lines)


class TestResumeHelp:
    def test_eval_help_lists_resume(self):
        result = CliRunner().invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--resume" in result.output

    def test_run_dataset_help_lists_resume(self):
        result = CliRunner().invoke(cli, ["run-dataset", "--help"])
        assert result.exit_code == 0
        assert "--resume" in result.output


class TestEvalResume:
    def test_resume_skips_done_workloads(self, tmp_path):
        kd = _write_kernel(tmp_path)
        out = tmp_path / "out.jsonl"
        append_jsonl_line(_passed_trace("w1"), out)
        append_jsonl_line(_passed_trace("w2"), out)

        fake, calls = _make_fake_runner()
        with patch("npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake):
            result = CliRunner().invoke(
                cli, ["eval", str(kd), "-o", str(out), "--resume"]
            )
        assert result.exit_code == 0, result.output
        # only the pending workload (w3) was evaluated
        assert calls == [["w3"]]
        # old (w1, w2) preserved + new (w3) appended => union
        assert _read_uuids(out) == ["w1", "w2", "w3"]

    def test_fresh_run_overwrites_output(self, tmp_path):
        kd = _write_kernel(tmp_path)
        out = tmp_path / "out.jsonl"
        append_jsonl_line(_passed_trace("stale"), out)  # unrelated prior data

        fake, calls = _make_fake_runner()
        with patch("npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake):
            result = CliRunner().invoke(
                cli, ["eval", str(kd), "-o", str(out)]
            )
        assert result.exit_code == 0, result.output
        assert calls == [["w1", "w2", "w3"]]
        uuids = _read_uuids(out)
        assert uuids == ["w1", "w2", "w3"]  # stale overwritten away
        assert "stale" not in uuids

    def test_all_done_skips_build_and_eval(self, tmp_path):
        kd = _write_kernel(tmp_path)
        out = tmp_path / "out.jsonl"
        for uuid in ("w1", "w2", "w3"):
            append_jsonl_line(_passed_trace(uuid), out)

        fake, calls = _make_fake_runner()
        with patch("npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake):
            result = CliRunner().invoke(
                cli, ["eval", str(kd), "-o", str(out), "--resume"]
            )
        assert result.exit_code == 0, result.output
        assert calls == []  # pre_build + eval skipped entirely
        assert _read_uuids(out) == ["w1", "w2", "w3"]  # untouched


def _write_dataset(base: Path) -> Path:
    """Write a data_dir with one kernel at level1/add/."""
    data_dir = base / "data"
    _write_kernel(data_dir / "level1", "add")
    return data_dir


class TestRunDatasetResume:
    def test_resume_skips_fully_done_kernel(self, tmp_path):
        data_dir = _write_dataset(tmp_path)
        out = tmp_path / "out.jsonl"
        for uuid in ("w1", "w2", "w3"):
            append_jsonl_line(_passed_trace(uuid), out)

        fake, calls = _make_fake_runner()
        with patch("npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake):
            result = CliRunner().invoke(
                cli, ["run-dataset", str(data_dir), "--level", "level1",
                      "-o", str(out), "--resume"]
            )
        assert result.exit_code == 0, result.output
        assert calls == []  # fully-done kernel skipped, no pre_build
        assert _read_uuids(out) == ["w1", "w2", "w3"]  # untouched

    def test_resume_runs_only_pending(self, tmp_path):
        data_dir = _write_dataset(tmp_path)
        out = tmp_path / "out.jsonl"
        append_jsonl_line(_passed_trace("w1"), out)  # w1 done

        fake, calls = _make_fake_runner()
        with patch("npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake):
            result = CliRunner().invoke(
                cli, ["run-dataset", str(data_dir), "--level", "level1",
                      "-o", str(out), "--resume"]
            )
        assert result.exit_code == 0, result.output
        assert calls == [["w2", "w3"]]  # only pending run
        assert _read_uuids(out) == ["w1", "w2", "w3"]  # union

    def test_resume_multi_kernel_loads_index_once(self, tmp_path):
        # Two kernels: 'add' (a1,a2,a3) fully done, 'sub' (s1,s2) partial.
        level1 = tmp_path / "data" / "level1"
        _write_named_kernel(level1, "add", ["a1", "a2", "a3"])
        _write_named_kernel(level1, "sub", ["s1", "s2"])
        data_dir = tmp_path / "data"
        out = tmp_path / "out.jsonl"
        for u in ("a1", "a2", "a3"):
            append_jsonl_line(_passed_trace(u, "add", "add_v1"), out)
        append_jsonl_line(_passed_trace("s1", "sub", "sub_v1"), out)

        fake, calls = _make_fake_runner()
        with patch(
            "npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake
        ), patch(
            "npu_kernelbench.cli.main._load_resume_index", wraps=load_resume_index
        ) as load_spy:
            result = CliRunner().invoke(
                cli, ["run-dataset", str(data_dir), "--level", "level1",
                      "-o", str(out), "--resume"]
            )
        assert result.exit_code == 0, result.output
        # Index loaded exactly once for the whole run, not per-kernel.
        assert load_spy.call_count == 1
        # 'add' fully done -> skipped; 'sub' -> only s2 pending.
        assert calls == [["s2"]]
        # Union preserved across kernels.
        assert _read_uuids(out) == ["a1", "a2", "a3", "s1", "s2"]

    def test_resume_aggregate_score_includes_reused(self, tmp_path):
        # Single fully-done kernel: nothing runs, but its reused traces must
        # still reach compute_score (aggregate reflects the full dataset).
        data_dir = tmp_path / "data"
        _write_named_kernel(data_dir / "level1", "add", ["a1", "a2", "a3"])
        out = tmp_path / "out.jsonl"
        for u in ("a1", "a2", "a3"):
            append_jsonl_line(_passed_trace(u, "add", "add_v1"), out)

        fake, calls = _make_fake_runner()
        seen: list[list[str]] = []

        def score_spy(traces):
            seen.append(sorted(t.workload.uuid for t in traces))
            return _real_compute_score(traces)

        with patch(
            "npu_kernelbench.cli.main._evaluate_with_runner", side_effect=fake
        ), patch(
            "npu_kernelbench.cli.main.compute_score", side_effect=score_spy
        ):
            result = CliRunner().invoke(
                cli, ["run-dataset", str(data_dir), "--level", "level1",
                      "-o", str(out), "--resume"]
            )
        assert result.exit_code == 0, result.output
        assert calls == []  # fully-done kernel, nothing run
        # The skipped kernel's REUSED traces still reach compute_score.
        assert ["a1", "a2", "a3"] in seen
