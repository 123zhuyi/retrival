from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

from triton_ascend_kb.cli import main


def run_json(capsys, argv):
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_source_to_wiki_to_query_index_workflow(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    assert main(["build-index", "--root", root]) == 0
    generated = capsys.readouterr().out
    assert "queries/by-technique.md" in generated
    assert "queries/by-kernel-type.md" in generated

    assert main(["validate", "--root", root]) == 0
    assert "OK" in capsys.readouterr().out

    pages = run_json(
        capsys,
        ["query", "--root", root, "rmsnorm", "vecutil_low", "--architecture", "Ascend910B", "--json"],
    )
    assert {page["id"] for page in pages} >= {"pr-triton-ascend-fixture-993", "kernel-rmsnorm-rope"}

    source = run_json(capsys, ["get-page", "--root", root, "pr-triton-ascend-fixture-993", "--json"])
    assert source["frontmatter"]["fixture"] is True
    assert "immutable source evidence" in source["body"]

    hits = run_json(capsys, ["grep", "--root", root, "pair_base", "--scope", "all", "--json"])
    assert hits


def test_agent_receives_kernelwiki_page_bundle(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    bundle = run_json(
        capsys,
        [
            "bundle",
            "--root",
            root,
            "rmsnorm",
            "vecutil_low",
            "--type",
            "kernel",
            "--architecture",
            "Ascend910B1",
            "--json",
        ],
    )
    assert bundle["retrieval"] == "kernelwiki_title_tag_body"
    assert [page["id"] for page in bundle["pages"]] == ["kernel-rmsnorm-rope"]
    assert {source["id"] for source in bundle["source_excerpts"]} == {
        "pr-triton-ascend-fixture-993",
        "run-rmsnorm-rope-001",
    }
    serialized = json.dumps(bundle, ensure_ascii=False)
    assert "query_type" not in serialized
    assert "condition_match_score" not in serialized
    assert "composite_score" not in serialized
    assert "Step 0" not in serialized


def test_codegen_debug_and_performance_queries_are_page_queries(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    cases = [
        (["elementwise", "tail_mask"], "kernel-elementwise-add"),
        (["rope", "pointer_rewrite", "compile_error"], "kernel-rope-pointer-addressing"),
        (["rmsnorm", "vecutil_low", "load_order"], "kernel-rmsnorm-rope"),
    ]
    for query, required_page in cases:
        bundle = run_json(capsys, ["bundle", "--root", root, *query, "--limit", "5", "--json"])
        assert required_page in {page["id"] for page in bundle["pages"]}
        assert bundle["source_excerpts"]


def test_human_readable_kernelwiki_cli_flow(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    assert main(["query", "--root", root, "tail_mask", "--type", "technique"]) == 0
    assert "technique-tail-mask" in capsys.readouterr().out

    assert main(["get-page", "--root", root, "kernel-elementwise-add"]) == 0
    assert "# Elementwise add" in capsys.readouterr().out

    assert main(["grep", "--root", root, "offsets < n", "--scope", "wiki"]) == 0
    assert "wiki/" in capsys.readouterr().out

    assert main(["bundle", "--root", root, "rope", "pointer_rewrite", "--limit", "2"]) == 0
    bundle_text = capsys.readouterr().out
    assert "KernelWiki Agent Knowledge Bundle" in bundle_text
    assert "Cited Source Excerpts" in bundle_text


def test_kernelwiki_native_scripts_contract(isolated_wiki) -> None:
    scripts = Path(__file__).parents[2] / "kernelwiki" / "scripts"
    env = os.environ | {"TRITON_ASCEND_KERNELWIKI_ROOT": str(isolated_wiki)}

    def run(script: str, *args: str) -> str:
        completed = subprocess.run(
            [sys.executable, str(scripts / script), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
        )
        return completed.stdout

    assert "kernel-rmsnorm-rope" in run("query.py", "rmsnorm", "--compact")
    page = run("get_page.py", "kernel-rmsnorm-rope", "--follow-sources", "--include-code")
    assert "pr-triton-ascend-fixture-993" in page
    assert "rmsnorm_rope.py" in page
    assert "rope-pointer-addressing.md" in run("grep_wiki.py", "pair_base", "--files-only")
    assert "KernelWiki validation passed" in run("validate.py")
    assert "Artifact bundles: 2" in run("repo_status.py")
    assert "by-hardware-feature.md" in run("generate-indices.py")
    assert not (isolated_wiki / "queries" / "by-language.md").exists()
    assert "changed=0" in run("retag_triton_ascend_docs.py")


def test_package_entry_covers_kernelwiki_native_modes(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    assert main(["query", "--root", root, "rmsnorm", "--compact"]) == 0
    assert "[wiki-kernel]" in capsys.readouterr().out
    assert main(["query", "--root", root, "rmsnorm", "--paths-only"]) == 0
    assert "rmsnorm-rope.md" in capsys.readouterr().out
    assert main(["query", "--root", root, "absent-term"]) == 0
    assert "No matching pages" in capsys.readouterr().out

    assert main(["get-page", "--root", root, "kernel-rmsnorm-rope", "--frontmatter-only"]) == 0
    assert "performance_claims" in capsys.readouterr().out
    assert main(["get-page", "--root", root, "kernel-rmsnorm-rope", "--body-only"]) == 0
    assert "# RMSNorm" in capsys.readouterr().out
    assert main(["get-page", "--root", root, "kernel-rmsnorm-rope", "--follow-sources", "--include-code"]) == 0
    assert "rmsnorm_rope.py" in capsys.readouterr().out

    assert main(["grep", "--root", root, "reference_rmsnorm_rope", "--only", "artifacts", "--files-only"]) == 0
    assert "rmsnorm_rope.py" in capsys.readouterr().out
    assert main(["repo-status", "--root", root]) == 0
    assert "Evaluation run pages: 3" in capsys.readouterr().out

    tagged = run_json(
        capsys,
        ["query", "--root", root, "--repo", "triton-lang/triton-ascend", "--tag", "attention", "--limit", "10", "--json"],
    )
    assert any("attention" in page["frontmatter"].get("kernel_types", []) for page in tagged)
    errors = run_json(
        capsys,
        ["query", "--root", root, "--repo", "triton-lang/triton-ascend", "--symptom", "ub_overflow", "--limit", "20", "--json"],
    )
    assert any(page["id"].endswith("debug-guide-ub-overflow") for page in errors)

    provenance = isolated_wiki / "artifacts" / "elementwise" / "PROVENANCE.yaml"
    data = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    data["files"][0]["sha256"] = "invalid"
    provenance.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert main(["validate", "--root", root]) == 1
    assert "sha256 mismatch" in capsys.readouterr().out


def test_upstream_doc_ingestion_is_pinned_and_idempotent(isolated_wiki, tmp_path) -> None:
    repo = tmp_path / "upstream"
    (repo / "docs" / "en").mkdir(parents=True)
    (repo / "docs" / "zh").mkdir(parents=True)
    (repo / "docs" / "en" / "debug_guide").mkdir(parents=True)
    (repo / "docs" / "zh" / "examples").mkdir(parents=True)
    (repo / "docs" / "en" / "debug_guide" / "ub_overflow.md").write_text(
        "# UB Overflow\n\nUBSIZE Out of Memory.\n", encoding="utf-8"
    )
    (repo / "docs" / "zh" / "examples" / "04_fused_attention_example.md").write_text(
        "# 融合注意力\n\nFused Attention on Ascend950PR (DAV_3510) with SIMT, L0C -> UB, and BufferID sync.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "docs"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "docs fixture"], check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    script = Path(__file__).parents[2] / "kernelwiki" / "scripts" / "ingest_triton_ascend_docs.py"
    env = os.environ | {"TRITON_ASCEND_KERNELWIKI_ROOT": str(isolated_wiki)}
    command = [sys.executable, str(script), "--repo-path", str(repo), "--retrieved-at", "2026-07-15", "--prune", "--allow-unverified-origin"]
    first = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", env=env).stdout
    second = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", env=env).stdout
    assert "Imported 2 document pages" in first
    assert "changed=0" in second
    manifest = yaml.safe_load((isolated_wiki / "data" / "triton-ascend-doc-snapshot.yaml").read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == commit
    assert manifest["page_count"] == 2
    attention_page = isolated_wiki / "sources" / "docs" / "triton-ascend" / "zh" / "examples" / "04_fused_attention_example.md"
    assert attention_page.is_file()
    attention_fm = yaml.safe_load(attention_page.read_text(encoding="utf-8").split("---", 2)[1])
    assert attention_fm["kernel_types"] == ["attention", "vfusion"]
    assert attention_fm["operators"] == ["fused-attention"]
    assert attention_fm["architectures"] == ["Ascend950PR"]
    assert attention_fm["hardware_features"] == [
        "l0c_buffer",
        "unified_buffer",
        "simt",
        "l0c_to_ub",
        "buffer_id_sync",
    ]
    ub_page = isolated_wiki / "sources" / "docs" / "triton-ascend" / "en" / "debug_guide" / "ub_overflow.md"
    ub_fm = yaml.safe_load(ub_page.read_text(encoding="utf-8").split("---", 2)[1])
    assert ub_fm["symptoms"] == ["ub_overflow", "out_of_memory"]
