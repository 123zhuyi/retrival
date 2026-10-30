from __future__ import annotations

import yaml

from triton_ascend_kb.cli import main
from triton_ascend_kb.indexer import INDEX_SPECS, build_indices, validate_corpus


def test_generate_kernelwiki_indices(isolated_wiki) -> None:
    paths = build_indices(isolated_wiki)
    assert [path.name for path in paths] == [spec[0] for spec in INDEX_SPECS]
    assert "kernel-rmsnorm-rope" in (isolated_wiki / "queries" / "by-kernel-type.md").read_text(encoding="utf-8")
    assert "load_order" in (isolated_wiki / "queries" / "by-technique.md").read_text(encoding="utf-8")
    assert validate_corpus(isolated_wiki) == []


def test_validator_reports_kernelwiki_contract_errors(isolated_wiki) -> None:
    broken = isolated_wiki / "wiki" / "patterns" / "broken.md"
    broken.write_text("---\nid: kernel-rmsnorm-rope\nsources: [missing]\n---\nbody", encoding="utf-8")
    errors = validate_corpus(isolated_wiki)
    assert any("duplicate page id" in error for error in errors)
    assert any("missing title" in error for error in errors)
    assert any("wiki page missing type" in error for error in errors)
    assert any("wiki page missing confidence" in error for error in errors)
    assert any("unresolved source" in error for error in errors)


def test_validator_rejects_legacy_sources_key(isolated_wiki) -> None:
    page = isolated_wiki / "wiki" / "patterns" / "legacy.md"
    page.write_text("---\nid: pattern-legacy\ntitle: Legacy\ntype: pattern\nsource_pages: []\n---\n", encoding="utf-8")
    assert any("legacy 'source_pages'" in error for error in validate_corpus(isolated_wiki))


def test_validator_rejects_removed_language_tags(isolated_wiki) -> None:
    page = isolated_wiki / "wiki" / "techniques" / "tail-mask.md"
    text = page.read_text(encoding="utf-8").replace(
        "architectures: [Ascend910B1]", "architectures: [Ascend910B1]\nlanguages: [triton_ascend]"
    )
    page.write_text(text, encoding="utf-8")
    assert any("legacy languages field" in error for error in validate_corpus(isolated_wiki))


def test_cli_query_bundle_validate_and_error(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    assert main(["query", "--root", root, "rmsnorm", "--json"]) == 0
    assert "kernel-rmsnorm-rope" in capsys.readouterr().out
    assert main(["bundle", "--root", root, "rmsnorm", "--type", "kernel", "--json"]) == 0
    assert "source_excerpts" in capsys.readouterr().out
    assert main(["validate", "--root", root]) == 0
    assert "KernelWiki validation passed" in capsys.readouterr().out
    assert main(["get-page", "--root", root, "missing"]) == 2
    assert "page not found" in capsys.readouterr().err
    assert main(["repo-status", "--root", root]) == 0
    assert "Artifact bundles" in capsys.readouterr().out


def test_cli_kernelwiki_compatible_output_modes(isolated_wiki, capsys) -> None:
    root = str(isolated_wiki)
    assert main(["query", "--root", root, "rmsnorm", "--paths-only"]) == 0
    assert "wiki/kernels/rmsnorm-rope.md" in capsys.readouterr().out
    assert main(["query", "--root", root, "rmsnorm", "--compact"]) == 0
    assert "[wiki-kernel]" in capsys.readouterr().out
    assert main(["query", "--root", root, "not-in-wiki"]) == 0
    assert "No matching pages" in capsys.readouterr().out
    assert main(["query", "--root", root, "tail_mask"]) == 0
    assert "**sources**" in capsys.readouterr().out

    assert main(["get-page", "--root", root, "kernel-rmsnorm-rope", "--frontmatter-only"]) == 0
    assert "confidence: experimental" in capsys.readouterr().out
    assert main(["get-page", "--root", root, "kernel-rmsnorm-rope", "--body-only"]) == 0
    assert "# RMSNorm" in capsys.readouterr().out
    assert main(["get-page", "--root", root, "kernel-rmsnorm-rope", "--follow-sources", "--include-code"]) == 0
    page = capsys.readouterr().out
    assert "Source: pr-triton-ascend-fixture-993" in page
    assert "rmsnorm_rope.py" in page

    assert main(["grep", "--root", root, "pair_base", "--files-only"]) == 0
    assert "rope-pointer-addressing.md" in capsys.readouterr().out
    assert main(["grep", "--root", root, "pair_base", "--json"]) == 0
    assert '"line_number"' in capsys.readouterr().out
    assert main(["grep", "--root", root, "reference_rmsnorm_rope", "--only", "artifacts", "--ext", "py"]) == 0
    assert "rmsnorm_rope.py" in capsys.readouterr().out
    assert main(["bundle", "--root", root, "rmsnorm", "--type", "kernel"]) == 0
    assert "Knowledge Bundle" in capsys.readouterr().out


def test_validator_checks_provenance_and_schema_failures(isolated_wiki, capsys) -> None:
    provenance = isolated_wiki / "artifacts" / "elementwise" / "PROVENANCE.yaml"
    data = yaml.safe_load(provenance.read_text(encoding="utf-8"))
    data["files"][0]["sha256"] = "bad"
    provenance.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    page = isolated_wiki / "wiki" / "techniques" / "tail-mask.md"
    text = page.read_text(encoding="utf-8").replace("confidence: experimental", "confidence: unknown")
    text = text.replace("reproducibility: snippet", "reproducibility: concept")
    text = text.replace("techniques: [tail_mask]", "techniques: [unknown_technique]")
    page.write_text(text, encoding="utf-8")
    errors = validate_corpus(isolated_wiki)
    assert any("sha256 mismatch" in error for error in errors)
    assert any("unknown confidence" in error for error in errors)
    assert any("reproducibility must be at least" in error for error in errors)
    assert any("unknown techniques" in error for error in errors)
    assert main(["validate", "--root", str(isolated_wiki)]) == 1
    assert "ERROR:" in capsys.readouterr().out


def test_validator_checks_upstream_doc_snapshot(isolated_wiki) -> None:
    manifest_path = isolated_wiki / "data" / "triton-ascend-doc-snapshot.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    first = manifest["pages"][0]
    manifest["page_count"] += 1
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    page_path = isolated_wiki / first["local_path"]
    page_path.write_text(page_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    errors = validate_corpus(isolated_wiki)
    assert any("page_count mismatch" in error for error in errors)
    assert any("snapshot body sha256 mismatch" in error for error in errors)


def test_validator_checks_controlled_doc_taxonomy(isolated_wiki) -> None:
    taxonomy_path = isolated_wiki / "data" / "operator-taxonomy-sources.yaml"
    taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy["sources"][0]["categories"].append("unknown-cann-category")
    taxonomy_path.write_text(yaml.safe_dump(taxonomy, sort_keys=False), encoding="utf-8")
    rules_path = isolated_wiki / "data" / "doc-tagging-rules.yaml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    rules["symptoms"].append({"value": "unknown-error", "body_regex": ["("]})
    rules["architectures"].append({"value": "unknown-arch", "body_regex": ["unknown"]})
    rules["hardware_features"].append({"value": "unknown-hardware", "body_regex": ["unknown"]})
    rules_path.write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")
    architecture_aliases_path = isolated_wiki / "data" / "architecture-aliases.yaml"
    architecture_aliases = yaml.safe_load(architecture_aliases_path.read_text(encoding="utf-8"))
    architecture_aliases["bad-selector"] = ["unknown-chip"]
    architecture_aliases_path.write_text(
        yaml.safe_dump(architecture_aliases, sort_keys=False), encoding="utf-8"
    )
    errors = validate_corpus(isolated_wiki)
    assert any("unknown kernel type unknown-cann-category" in error for error in errors)
    assert any("unknown symptoms value unknown-error" in error for error in errors)
    assert any("unknown architectures value unknown-arch" in error for error in errors)
    assert any("unknown hardware_features value unknown-hardware" in error for error in errors)
    assert any("unknown chip model unknown-chip" in error for error in errors)
    assert any("invalid regex" in error for error in errors)
