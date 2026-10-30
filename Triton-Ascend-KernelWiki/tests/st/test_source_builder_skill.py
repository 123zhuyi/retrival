from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from triton_ascend_kb.indexer import validate_corpus
from triton_ascend_kb.kernelwiki import KernelWikiSearch, SearchFilters
from triton_ascend_kb.source_builder import main as source_builder_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = (
    PROJECT_ROOT
    / "integrations"
    / "cannbot-skills"
    / "ops"
    / "triton-kb-source-builder"
)
PACKET = PROJECT_ROOT / "examples" / "source-builder" / "dynamic-tail-mask.yaml"


def test_source_builder_skill_publishes_reviewed_pages_end_to_end(
    isolated_wiki: Path, tmp_path: Path, capsys
) -> None:
    env = os.environ | {"TRITON_ASCEND_KB_ROOT": str(isolated_wiki)}
    completed = subprocess.run(
        [
            sys.executable,
            str(SKILL_ROOT / "scripts" / "build.py"),
            "--root",
            str(isolated_wiki),
            "--workdir",
            str(tmp_path / "builds"),
            "run",
            "--packet",
            str(PACKET),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    wrapper_result = json.loads(completed.stdout)
    assert wrapper_result["verify"]["decision"] == "ready-for-review"

    assert source_builder_main(
        [
            "--root",
            str(isolated_wiki),
            "--workdir",
            str(tmp_path / "direct-builds"),
            "run",
            "--packet",
            str(PACKET),
            "--publish",
            "--approved-by",
            "demo-reviewer",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["publish"]["decision"] == "published"
    assert result["publish"]["gates"]["human_approval"] is True
    assert (isolated_wiki / "sources" / "prs" / "source-builder-demo" / "PR-1200.md").is_file()
    assert (isolated_wiki / "wiki" / "techniques" / "dynamic-extent-tail-mask.md").is_file()
    assert validate_corpus(isolated_wiki) == []

    pages = KernelWikiSearch(isolated_wiki).search(
        ["dynamic", "tail_mask"],
        filters=SearchFilters(type="technique", architecture="A2"),
        limit=5,
    )
    assert "technique-dynamic-extent-tail-mask" in {page.id for page in pages}
    index = (isolated_wiki / "queries" / "by-technique.md").read_text(encoding="utf-8")
    assert "technique-dynamic-extent-tail-mask" in index


def test_source_builder_skill_metadata_and_policy_are_installable() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    metadata = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    schema = (SKILL_ROOT / "references" / "source-packet-schema.md").read_text(encoding="utf-8")
    governance = (SKILL_ROOT / "references" / "tag-governance.md").read_text(encoding="utf-8")
    assert "name: triton-kb-source-builder" in skill
    assert "collect`, `build`, `verify`, and `publish" in skill
    assert "Never edit `tags.yaml`" in skill
    assert "$triton-kb-source-builder" in metadata
    assert "source_type" in schema
    assert "npu-hardware-params.md" in governance
