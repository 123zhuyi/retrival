from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from triton_ascend_kb.cannbot_retriever import main as retriever_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = (
    PROJECT_ROOT
    / "integrations"
    / "cannbot-skills"
    / "ops"
    / "triton-kernelwiki-retriever"
)


def run_skill(tmp_path: Path, wiki_root: Path, *args: str) -> dict:
    env = os.environ | {"TRITON_ASCEND_KB_ROOT": str(wiki_root)}
    completed = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts" / "retrieve.py"),
            "--workdir",
            str(tmp_path),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    output = Path(completed.stdout.strip())
    assert output.is_file()
    return json.loads(output.read_text(encoding="utf-8"))


def test_skill_design_repair_and_optimize_workflow(tmp_path: Path, wiki_root: Path) -> None:
    design = run_skill(
        tmp_path,
        wiki_root,
        "--mode",
        "design",
        "--op-name",
        "LayerNorm",
        "--semantic-op",
        "layer_norm",
        "--category",
        "normalization",
        "--arch",
        "ascend910b1",
        "--limit",
        "1",
    )
    assert design["mode"] == "design"
    assert design["retrieval"]["second_stage_reranking"] is False
    official = next(lane for lane in design["lanes"] if lane["name"] == "official")
    assert official["bundle"]["pages"][0]["id"].endswith("03-layer-norm-example")

    error_file = tmp_path / "verifier_error.txt"
    error_file.write_text("allclose failed: relative error too large", encoding="utf-8")
    repair = run_skill(
        tmp_path,
        wiki_root,
        "--mode",
        "repair",
        "--op-name",
        "RMSNorm",
        "--kernel-type",
        "norm",
        "--keyword",
        "fp32_accumulation",
        "--verifier-error-file",
        str(error_file),
        "--iteration",
        "1",
        "--limit",
        "1",
    )
    assert repair["request"]["symptom"] == "precision_mismatch"
    assert (tmp_path / "knowledge" / "repair_iter_1.json").is_file()

    optimize = run_skill(
        tmp_path,
        wiki_root,
        "--mode",
        "optimize",
        "--op-name",
        "RMSNorm",
        "--kernel-type",
        "norm",
        "--keyword",
        "load_order",
        "--iteration",
        "0",
        "--limit",
        "1",
    )
    assert optimize["request"]["symptom"] == "performance_bottleneck"
    assert (tmp_path / "knowledge" / "optimize_iter_0.json").is_file()


def test_skill_metadata_and_contract_are_installable() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    metadata = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    contract = (SKILL_ROOT / "references" / "integration-contract.md").read_text(encoding="utf-8")
    assert "name: triton-kernelwiki-retriever" in skill
    assert "Do not write successful outputs back to KernelWiki" in skill
    assert "$triton-kernelwiki-retriever" in metadata
    assert "second_stage_reranking" in contract


def test_installed_entrypoint_executes_all_orchestrator_modes(
    tmp_path: Path, wiki_root: Path, capsys
) -> None:
    common = ["--root", str(wiki_root), "--workdir", str(tmp_path), "--limit", "1"]
    assert retriever_main(
        [
            "--mode",
            "design",
            *common,
            "--op-name",
            "LayerNorm",
            "--semantic-op",
            "layer_norm",
            "--category",
            "normalization",
        ]
    ) == 0
    design_path = Path(capsys.readouterr().out.strip())
    assert json.loads(design_path.read_text(encoding="utf-8"))["mode"] == "design"

    assert retriever_main(
        [
            "--mode",
            "repair",
            *common,
            "--op-name",
            "RMSNorm",
            "--kernel-type",
            "norm",
            "--symptom",
            "precision_mismatch",
            "--keyword",
            "fp32_accumulation",
            "--iteration",
            "2",
        ]
    ) == 0
    repair_path = Path(capsys.readouterr().out.strip())
    assert json.loads(repair_path.read_text(encoding="utf-8"))["mode"] == "repair"

    assert retriever_main(
        [
            "--mode",
            "optimize",
            *common,
            "--op-name",
            "RMSNorm",
            "--kernel-type",
            "norm",
            "--keyword",
            "load_order",
            "--iteration",
            "2",
        ]
    ) == 0
    optimize_path = Path(capsys.readouterr().out.strip())
    assert json.loads(optimize_path.read_text(encoding="utf-8"))["mode"] == "optimize"
