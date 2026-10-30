"""Tests for the external Level3 solution preparation tool."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "prepare_level3_solutions.py"


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _write_fixture(
    tmp_path: Path,
    bad_hash: bool = False,
) -> tuple[Path, Path, Path, Path]:
    source_root = tmp_path / "sources"
    data_root = tmp_path / "data" / "level3"
    output_root = tmp_path / "solutions"
    source_root.mkdir()

    sources = {
        "Positional": (
            "1_Positional_generated.py",
            "class ModelNew:\n    def forward(self, A, B):\n        return A\n",
            ["A", "B"],
            "positional",
        ),
        "Packed": (
            "2_Packed_generated.py",
            "class ModelNew:\n    def forward(self, inputs):\n        return inputs[0]\n",
            ["x", "attr"],
            "packed_inputs",
        ),
    }

    operators = []
    for definition, (filename, source, inputs, call_style) in sources.items():
        payload = source.replace("\n", "\r\n") if definition == "Packed" else source
        (source_root / filename).write_bytes(payload.encode("utf-8"))
        definition_dir = data_root / definition
        definition_dir.mkdir(parents=True)
        (definition_dir / "definition.json").write_text(
            json.dumps({"inputs": {name: {} for name in inputs}}),
            encoding="utf-8",
        )
        operators.append(
            {
                "definition": definition,
                "source": filename,
                "sha256": "0" * 64 if bad_hash else _sha256(source),
                "call_style": call_style,
            }
        )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "forbidden_markers": ["tl.compile_hint"],
                "operators": operators,
                "known_missing": [],
            }
        ),
        encoding="utf-8",
    )
    return source_root, data_root, output_root, manifest


def _run_tool(
    source_root: Path,
    data_root: Path,
    output_root: Path,
    manifest: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--data-root",
            str(data_root),
            "--output-root",
            str(output_root),
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_builds_positional_and_packed_input_wrappers(tmp_path: Path):
    source_root, data_root, output_root, manifest = _write_fixture(tmp_path)
    result = _run_tool(source_root, data_root, output_root, manifest)

    assert result.returncode == 0, result.stdout + result.stderr
    positional = json.loads(
        (output_root / "level3" / "Positional" / "solution.json").read_text()
    )
    packed = json.loads(
        (output_root / "level3" / "Packed" / "solution.json").read_text()
    )
    positional_source = positional["sources"][0]["content"]
    packed_source = packed["sources"][0]["content"]
    ast.parse(positional_source)
    ast.parse(packed_source)
    assert "def run(A, B):" in positional_source
    assert "return _MODEL(A, B)" in positional_source
    assert "def run(x, attr):" in packed_source
    assert "return _MODEL((x, attr,))" in packed_source


def test_rejects_source_hash_mismatch(tmp_path: Path):
    source_root, data_root, output_root, manifest = _write_fixture(
        tmp_path,
        bad_hash=True,
    )
    result = _run_tool(source_root, data_root, output_root, manifest)

    assert result.returncode == 1
    assert "SHA-256 mismatch" in result.stdout
    assert not output_root.exists()
