"""Integration tests: full CLI → eval → traces pipeline."""

import pytest
from pathlib import Path

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except (ImportError, RuntimeError):
    HAS_NPU = False

from click.testing import CliRunner
from npu_kernelbench.cli.main import cli


SAMPLES = Path(__file__).parent / "samples"


class TestCliErrors:
    def test_non_existent_problem_dir(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "/nonexistent/path", "--solution", "/tmp/x.json"])
        assert result.exit_code != 0


@pytest.mark.skipif(not HAS_NPU, reason="NPU hardware required")
class TestIntegration:
    def test_identity_kernel_e2e(self, tmp_path):
        """Run identity kernel end-to-end through the CLI."""
        import json

        sol_path = tmp_path / "solution.json"
        sol_path.write_text(json.dumps({
            "name": "identity_sol",
            "definition": "identity",
            "author": "test",
            "spec": {
                "languages": ["tilelang"],
                "target_hardware": ["ascend910b"],
                "entry_point": "main.py::run",
            },
            "sources": [
                {"path": "main.py", "content": "def run(x): return x + 0"},
            ],
        }))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "eval",
                "--definition", str(SAMPLES / "identity" / "definition.json"),
                "--workload", str(SAMPLES / "identity" / "workload.jsonl"),
                "--solution", str(sol_path),
            ],
        )
        if result.exit_code != 0:
            print("STDOUT:", result.output)
            if result.exception:
                raise result.exception
        assert result.exit_code == 0
        assert "2/2" in result.output
        assert "PASSED" in result.output
