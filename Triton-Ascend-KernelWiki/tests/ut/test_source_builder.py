from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from triton_ascend_kb.io import read_markdown
from triton_ascend_kb.source_builder import SourceBuildError, SourceBuilder, main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKET = PROJECT_ROOT / "examples" / "source-builder" / "dynamic-tail-mask.yaml"


def write_packet(tmp_path: Path, mutate=None) -> Path:
    data = yaml.safe_load(PACKET.read_text(encoding="utf-8"))
    if mutate:
        mutate(data)
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_source_builder_collect_build_verify_is_idempotent(isolated_wiki: Path, tmp_path: Path) -> None:
    builder = SourceBuilder(isolated_wiki, tmp_path / "builds")
    first = builder.run(PACKET)
    second = builder.run(PACKET)

    assert first["verify"]["decision"] == "ready-for-review"
    assert first["verify"]["gates"]["human_approval"] is False
    assert first["build"]["tag_proposal_count"] == 0
    assert first["collect"]["packet_sha256"] == second["collect"]["packet_sha256"]
    build_dir = builder.build_dir("dynamic-tail-mask-demo")
    manifest = yaml.safe_load((build_dir / "evidence-manifest.yaml").read_text(encoding="utf-8"))
    assert len(manifest["evidence"]) == 2
    assert all(len(record["sha256"]) == 64 for record in manifest["evidence"])
    source = (build_dir / "candidates" / "source.md").read_text(encoding="utf-8")
    wiki = read_markdown(build_dir / "candidates" / "wiki.md")
    assert "pr-source-builder-fixture-1200" in source
    assert wiki.frontmatter["sources"] == ["pr-source-builder-fixture-1200"]


def test_unknown_controlled_tag_emits_proposal_and_blocks_publish(
    isolated_wiki: Path, tmp_path: Path
) -> None:
    packet = write_packet(
        tmp_path,
        lambda data: data["candidate"]["frontmatter"]["symptoms"].append("new-error-class"),
    )
    builder = SourceBuilder(isolated_wiki, tmp_path / "builds")
    result = builder.run(packet)

    assert result["verify"]["decision"] == "blocked"
    assert result["verify"]["gates"]["controlled_tags_valid"] is False
    proposal = result["verify"]["tag_proposals"][0]
    assert proposal["field"] == "symptoms"
    assert proposal["value"] == "new-error-class"
    assert proposal["status"] == "needs-taxonomy-review"
    assert "verifier-error taxonomy" in proposal["authority"]
    with pytest.raises(SourceBuildError, match="ready-for-review"):
        builder.publish("dynamic-tail-mask-demo", approved_by="reviewer")


def test_evidence_integrity_and_source_link_are_hard_gates(
    isolated_wiki: Path, tmp_path: Path
) -> None:
    builder = SourceBuilder(isolated_wiki, tmp_path / "builds")
    builder.collect(PACKET)
    builder.build("dynamic-tail-mask-demo")
    build_dir = builder.build_dir("dynamic-tail-mask-demo")
    (build_dir / "evidence" / "fixture-diff.txt").write_text("tampered", encoding="utf-8")
    wiki_path = build_dir / "candidates" / "wiki.md"
    wiki_path.write_text(
        wiki_path.read_text(encoding="utf-8").replace(
            "sources:\n- pr-source-builder-fixture-1200", "sources:\n- unrelated-source"
        ),
        encoding="utf-8",
    )

    report = builder.verify("dynamic-tail-mask-demo")
    assert report["decision"] == "blocked"
    assert report["gates"]["evidence_integrity"] is False
    assert report["gates"]["source_linked"] is False
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_packet_and_target_validation_errors(isolated_wiki: Path, tmp_path: Path) -> None:
    builder = SourceBuilder(isolated_wiki, tmp_path / "builds")
    bad_schema = write_packet(tmp_path, lambda data: data.update(schema_version=2))
    with pytest.raises(SourceBuildError, match="schema_version"):
        builder.collect(bad_schema)

    escaping = write_packet(
        tmp_path,
        lambda data: data["candidate"].update(target_path="../outside.md"),
    )
    builder.collect(escaping)
    with pytest.raises(SourceBuildError, match="stay below wiki"):
        builder.build("dynamic-tail-mask-demo")


def test_local_evidence_file_and_cli_stage_flow(isolated_wiki: Path, tmp_path: Path, capsys) -> None:
    evidence = tmp_path / "change.patch"
    evidence.write_text("+ mask = offsets < n\n", encoding="utf-8")

    def mutate(data):
        data["build_id"] = "local-evidence-demo"
        data["evidence"] = [
            {
                "id": "local-patch",
                "kind": "diff",
                "path": "change.patch",
                "summary": "Local evidence path branch",
            }
        ]

    packet = write_packet(tmp_path, mutate)
    root = str(isolated_wiki)
    workdir = str(tmp_path / "cli-builds")
    assert main(["--root", root, "--workdir", workdir, "collect", "--packet", str(packet)]) == 0
    assert json.loads(capsys.readouterr().out)["evidence_count"] == 1
    assert main(["--root", root, "--workdir", workdir, "build", "--build-id", "local-evidence-demo"]) == 0
    assert json.loads(capsys.readouterr().out)["tag_proposal_count"] == 0
    assert main(["--root", root, "--workdir", workdir, "verify", "--build-id", "local-evidence-demo"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "ready-for-review"


def test_cli_reports_missing_build_and_publish_approval(isolated_wiki: Path, tmp_path: Path, capsys) -> None:
    common = ["--root", str(isolated_wiki), "--workdir", str(tmp_path / "builds")]
    assert main([*common, "build", "--build-id", "missing-build"]) == 2
    assert "has not been collected" in capsys.readouterr().err

    builder = SourceBuilder(isolated_wiki, tmp_path / "builds")
    builder.run(PACKET)
    with pytest.raises(SourceBuildError, match="approved_by"):
        builder.publish("dynamic-tail-mask-demo", approved_by="")
