"""Test that compile failures produce COMPILE_ERROR traces in traces.jsonl."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from npu_kernelbench.data import (
    Definition,
    Workload,
    Solution,
    Trace,
    EvaluationStatus,
    load_jsonl_file,
)
from npu_kernelbench.cli.config import BenchmarkConfig
from npu_kernelbench.builder.builder import BuildError


MINIMAL_DEFINITION = {
    "name": "compile_fail_test",
    "axes": {"N": {"type": "var", "description": "batch size"}},
    "inputs": {
        "x": {"shape": ["N"], "dtype": "float32", "description": "input tensor"}
    },
    "outputs": {
        "y": {"shape": ["N"], "dtype": "float32", "description": "output tensor"}
    },
    "reference": "def run(x):\n    return x\n",
}


def _make_workload(uuid: str) -> Workload:
    return Workload(
        uuid=uuid,
        axes={"N": 4},
        inputs={"x": {"type": "random", "dtype": "float32"}},
    )


def _make_solution() -> Solution:
    return Solution(
        name="compile_fail_sol",
        definition="compile_fail_test",
        author="test",
        spec={
            "languages": ["python"],
            "target_hardware": ["ascend910b"],
            "entry_point": "main.py::run",
        },
        sources=[{"path": "main.py", "content": "def run(x): return x + 0"}],
    )


class TestCompileFailureTraces:
    """Compile failure in pre_build must produce COMPILE_ERROR traces + write JSONL."""

    def test_pre_build_builderror_produces_compile_error_traces(self, tmp_path):
        """When Runner.pre_build raises BuildError, one COMPILE_ERROR trace
        per workload is returned and written to output_file."""
        from npu_kernelbench.cli.main import _evaluate_with_runner

        definition = Definition(**MINIMAL_DEFINITION)
        workloads = [_make_workload("wl-1"), _make_workload("wl-2")]
        solution = _make_solution()
        config = BenchmarkConfig(runner="isolated")
        output_file = tmp_path / "traces.jsonl"

        # Mock IsolatedRunner to raise BuildError in pre_build
        mock_runner_cls = MagicMock()
        mock_runner_instance = mock_runner_cls.return_value.__enter__.return_value
        mock_runner_instance.pre_build.side_effect = BuildError("compile error", builder_name="ascendc")

        with patch.dict(
            "npu_kernelbench.cli.main.__dict__",
            {"IsolatedRunner": mock_runner_cls},
        ):
            traces = _evaluate_with_runner(
                definition, workloads, solution, config, output_file=output_file,
            )

        # Must return one trace per workload
        assert len(traces) == 2
        for t in traces:
            assert isinstance(t, Trace)
            assert t.evaluation is not None
            assert t.evaluation.status == EvaluationStatus.COMPILE_ERROR
            assert "compile error" in t.evaluation.log

        # Output file must exist with both traces
        assert output_file.exists()
        loaded = load_jsonl_file(Trace, output_file)
        assert len(loaded) == 2
        for t in loaded:
            assert t.evaluation.status == EvaluationStatus.COMPILE_ERROR
