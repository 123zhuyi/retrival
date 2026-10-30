from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_ascend_kb.agent_bundle import KernelWikiAgentAdapter
from triton_ascend_kb.cannbot_retriever import (
    _dedupe_aliases,
    _normalize_op_name,
    _unique_nonempty,
    build_context,
    default_output,
    infer_symptom,
    main,
    normalize_architecture,
    resolve_kernel_type,
    write_context,
)
from triton_ascend_kb.kernelwiki import KernelWikiSearch, SearchFilters


def test_normalization_category_mapping_and_alias_deduplication(wiki_root: Path) -> None:
    assert normalize_architecture(None) is None
    assert normalize_architecture(" Ascend910B1 ") == "Ascend910B1"
    assert normalize_architecture("A3") == "A3"
    assert normalize_architecture("arch35") == "DAV_3510"
    assert normalize_architecture("DAV_3113") == "DAV_3113"
    assert normalize_architecture("future-arch") == "future-arch"
    assert resolve_kernel_type(None, " Attention ") == "attention"
    assert resolve_kernel_type("normalization", None) == "norm"
    assert resolve_kernel_type("transformer-inference", None) is None
    assert _unique_nonempty([" LayerNorm ", "", None, "layernorm", "layer_norm"]) == [
        "LayerNorm",
        "layer_norm",
    ]
    adapter = KernelWikiAgentAdapter(wiki_root)
    assert _dedupe_aliases(adapter, ["LayerNorm", "layer_norm", "fp32_accumulation"]) == [
        "LayerNorm",
        "fp32_accumulation",
    ]


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("illegal memory access and runtime error", "illegal_memory_access"),
        ("UBSIZE exceeded before out of memory", "ub_overflow"),
        ("NPU out of memory", "out_of_memory"),
        ("Tensor shape mismatch", "shape_mismatch"),
        ("allclose failed with relative error", "precision_mismatch"),
        ("unsupported operation", "unsupported_operation"),
        ("misaligned address", "alignment_error"),
        ("MLIR compilation failed", "compile_error"),
        ("RuntimeError: device error", "runtime_error"),
        ("unclassified failure", None),
    ],
)
def test_infer_symptom(message: str, expected: str | None) -> None:
    assert infer_symptom(message) == expected


def test_design_context_preserves_independent_kernelwiki_lanes(wiki_root: Path) -> None:
    context = build_context(
        root=wiki_root,
        mode="design",
        op_name="LayerNorm",
        semantic_ops=["layer_norm"],
        category="normalization",
        architecture="ascend910b1",
        limit=1,
    )
    assert context["request"]["kernel_type"] == "norm"
    assert context["request"]["normalized_architecture"] == "Ascend910B1"
    assert context["retrieval"] == {
        "engine": "kernelwiki_title_tag_body",
        "second_stage_reranking": False,
        "limit_per_lane": 1,
        "lanes_are_independent": True,
    }
    lanes = {lane["name"]: lane for lane in context["lanes"]}
    assert set(lanes) == {"official", "wiki-kernel", "wiki-technique", "hardware"}
    assert lanes["official"]["bundle"]["query"] == ["LayerNorm"]
    assert lanes["official"]["bundle"]["pages"][0]["type"] == "source-doc"
    assert lanes["hardware"]["filters"] == {"type": "hardware", "architecture": "Ascend910B1"}
    assert not any("second ranking" in warning for warning in context["warnings"])


def test_design_context_warns_for_broad_category_and_empty_lanes(wiki_root: Path) -> None:
    context = build_context(
        root=wiki_root,
        mode="design",
        op_name="UnknownDetectionOperator",
        category="object-detection",
        architecture=None,
        limit=1,
    )
    assert len(context["lanes"]) == 3
    assert context["request"]["kernel_type"] is None
    assert any("no unambiguous CANN kernel_type" in warning for warning in context["warnings"])
    assert any("No matching pages" in warning for warning in context["warnings"])


def test_repair_and_optimize_contexts_use_controlled_symptoms(wiki_root: Path) -> None:
    repair = build_context(
        root=wiki_root,
        mode="repair",
        op_name="RMSNorm",
        category="normalization",
        verifier_error="allclose failed with relative error 0.1",
        keywords=["fp32_accumulation"],
        limit=2,
    )
    assert repair["request"]["symptom"] == "precision_mismatch"
    assert {lane["consumer_field"] for lane in repair["lanes"]} == {"repair_knowledge_context"}
    technique_lane = next(lane for lane in repair["lanes"] if lane["name"] == "wiki-technique-diagnostic")
    assert technique_lane["bundle"]["pages"][0]["id"] == "technique-fp32-accumulation"

    optimize = build_context(
        root=wiki_root,
        mode="optimize",
        op_name="RMSNorm",
        kernel_type="norm",
        symptom="runtime_error",
        keywords=["load_order"],
        limit=1,
    )
    assert optimize["request"]["symptom"] == "performance_bottleneck"
    assert {lane["consumer_field"] for lane in optimize["lanes"]} == {
        "optimization_knowledge_context"
    }


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"mode": "unknown", "op_name": "x"}, ValueError, "unsupported mode"),
        ({"mode": "design", "op_name": " "}, ValueError, "op_name"),
        ({"mode": "design", "op_name": "x", "limit": 0}, ValueError, "limit"),
        (
            {"mode": "repair", "op_name": "x", "verifier_error": "unclassified"},
            ValueError,
            "recognizable verifier error",
        ),
        ({"mode": "repair", "op_name": "x", "symptom": "made_up"}, ValueError, "unknown symptom"),
    ],
)
def test_build_context_rejects_invalid_requests(
    wiki_root: Path, kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build_context(root=wiki_root, **kwargs)


def test_build_context_rejects_non_kernelwiki_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a KernelWiki root"):
        build_context(root=tmp_path, mode="design", op_name="x")


def test_default_output_and_atomic_writer(tmp_path: Path) -> None:
    assert default_output(tmp_path, "design", 7).name == "design_context.json"
    assert default_output(tmp_path, "repair", 7).name == "repair_iter_7.json"
    assert default_output(tmp_path, "optimize", 7).name == "optimize_iter_7.json"
    output = write_context({"answer": "知识"}, tmp_path / "nested" / "context.json")
    assert json.loads(output.read_text(encoding="utf-8")) == {"answer": "知识"}
    assert not output.with_suffix(".json.tmp").exists()


def test_cli_reads_verifier_error_file_and_writes_default_output(
    wiki_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    error_file = tmp_path / "verifier_error.txt"
    error_file.write_text("output mismatch: allclose failed", encoding="utf-8")
    assert (
        main(
            [
                "--mode",
                "repair",
                "--root",
                str(wiki_root),
                "--workdir",
                str(tmp_path),
                "--op-name",
                "RMSNorm",
                "--category",
                "normalization",
                "--verifier-error-file",
                str(error_file),
                "--iteration",
                "3",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    output = Path(capsys.readouterr().out.strip())
    assert output.name == "repair_iter_3.json"
    assert json.loads(output.read_text(encoding="utf-8"))["request"]["symptom"] == "precision_mismatch"


def test_cli_requires_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRITON_ASCEND_KB_ROOT", raising=False)
    with pytest.raises(SystemExit, match="--root or TRITON_ASCEND_KB_ROOT"):
        main(["--mode", "design", "--workdir", str(tmp_path), "--op-name", "x"])


# ---------------------------------------------------------------------------
# Regression tests for retrieval recall gaps found via 4_MoeComputeExpertTokens:
#   - operators field not searched
#   - numeric prefix in benchmark filenames poisoning the query
#   - category->tag mismatch with no fallback
#   - architecture filter not doing family (sibling SKU) matching
# ---------------------------------------------------------------------------


def test_normalize_op_name_strips_numeric_prefix() -> None:
    assert _normalize_op_name("4_MoeComputeExpertTokens") == "MoeComputeExpertTokens"
    assert _normalize_op_name("12_FusedInferAttentionScore") == "FusedInferAttentionScore"
    assert _normalize_op_name("MoeComputeExpertTokens") == "MoeComputeExpertTokens"
    assert _normalize_op_name("  7_SparseFlashAttention ") == "SparseFlashAttention"
    # Non-numeric prefixes are preserved.
    assert _normalize_op_name("v2_MoeInitRouting") == "v2_MoeInitRouting"


def test_op_name_numeric_prefix_preserved_in_request_but_stripped_from_query(
    wiki_root: Path,
) -> None:
    context = build_context(
        root=wiki_root,
        mode="design",
        op_name="4_MoeComputeExpertTokens",
        semantic_ops=["moe_compute_expert_tokens"],
        category="index-computation",
        architecture="ascend910b3",
        limit=3,
    )
    # Original op_name preserved for traceability.
    assert context["request"]["op_name"] == "4_MoeComputeExpertTokens"
    # No query term carries the "4_" prefix.
    for lane in context["lanes"]:
        for term in lane["bundle"]["query"]:
            assert not term.startswith("4_")


def test_operators_field_searchable_by_canonical_op_name(wiki_root: Path) -> None:
    """The canonical op name lives in frontmatter `operators:`; searching by it must
    hit the page (previously TAG_FIELDS excluded `operators` and body excluded
    frontmatter, so the underscore op name scored 0)."""
    search = KernelWikiSearch(wiki_root)
    pages = search.search(
        ["moe_compute_expert_tokens"],
        filters=SearchFilters(type="kernel"),
        limit=5,
    )
    paths = [page.path for page in pages]
    assert "wiki/kernels/moe-compute-expert-tokens.md" in paths


def test_tag_filter_fallback_recovers_mismatched_category(wiki_root: Path) -> None:
    """category=index-computation maps to tag=index, which excludes the moe-tagged
    page. The lane must fall back to a no-tag search and mark it with `fallback`."""
    context = build_context(
        root=wiki_root,
        mode="design",
        op_name="4_MoeComputeExpertTokens",
        semantic_ops=["moe_compute_expert_tokens"],
        category="index-computation",
        architecture="ascend910b3",
        limit=3,
    )
    lanes = {lane["name"]: lane for lane in context["lanes"]}
    wiki_kernel = lanes["wiki-kernel"]
    assert wiki_kernel["page_count"] >= 1
    assert "tag" in wiki_kernel.get("fallback", "")
    titles = [page["title"] for page in wiki_kernel["bundle"]["pages"]]
    assert any("MoeComputeExpertTokens" in title for title in titles)


def test_architecture_family_reverse_lookup_matches_sibling_sku(wiki_root: Path) -> None:
    """Querying a specific SKU (Ascend910B3) must match pages tagged with a sibling
    SKU (Ascend910B1) of the same DAV_2201 family, via reverse lookup -- not via the
    architecture fallback."""
    search = KernelWikiSearch(wiki_root)
    expanded = {value.lower() for value in search.expand_architecture("Ascend910B3")}
    assert "ascend910b3" in expanded
    assert "ascend910b1" in expanded
    assert "ascend910b2" in expanded

    context = build_context(
        root=wiki_root,
        mode="design",
        op_name="4_MoeComputeExpertTokens",
        semantic_ops=["moe_compute_expert_tokens"],
        category="index-computation",
        architecture="ascend910b3",
        limit=3,
    )
    lanes = {lane["name"]: lane for lane in context["lanes"]}
    # hardware lane finds a page via family matching, so architecture fallback
    # must NOT be the reason (no "architecture" in the fallback marker).
    assert lanes["hardware"]["page_count"] >= 1
    assert "architecture" not in lanes["hardware"].get("fallback", "")
