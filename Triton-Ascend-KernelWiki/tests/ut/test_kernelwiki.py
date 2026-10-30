from __future__ import annotations

import pytest

from triton_ascend_kb.kernelwiki import KernelWikiSearch, SearchFilters


def test_alias_expansion_and_kernelwiki_score(wiki_root) -> None:
    search = KernelWikiSearch(wiki_root)
    assert search.expand_keyword("910b") == [
        "910b",
        "Ascend910B1",
        "Ascend910B2",
        "Ascend910B2C",
        "Ascend910B3",
        "Ascend910B4",
        "Ascend910_93",
    ]
    assert search.expand_keyword("Ascend910B1") == ["Ascend910B1"]
    assert search.expand_keyword("A5") == ["A5", "Ascend950DT", "Ascend950PR"]
    assert search.expand_keyword("unknown") == ["unknown"]
    pages = search.search(["rmsnorm", "vecutil_low"])
    assert pages[0].id in {"pr-triton-ascend-fixture-993", "kernel-rmsnorm-rope"}
    assert pages[0].score >= pages[-1].score > 0
    assert {page.id for page in pages} >= {"pr-triton-ascend-fixture-993", "kernel-rmsnorm-rope"}
    assert search.search(["not-in-wiki"]) == []


def test_kernelwiki_filters_and_has_code(wiki_root) -> None:
    search = KernelWikiSearch(wiki_root)
    filters = SearchFilters(
        type="kernel",
        tag="rms_norm",
        architecture="A2",
        confidence="experimental",
        has_code=True,
    )
    assert [page.id for page in search.search([], filters=filters)] == ["kernel-rmsnorm-rope"]
    assert search.search([], filters=SearchFilters(type="pr", repo="AscendOpGenAgent"))[0].id == "pr-triton-ascend-fixture-993"
    assert search.search([], filters=SearchFilters(symptom="compile_error"))
    assert search.search([], filters=SearchFilters(type="pattern", has_code=True)) == []


def test_get_page_and_grep(wiki_root) -> None:
    search = KernelWikiSearch(wiki_root)
    assert search.get_page("kernel-rmsnorm-rope").frontmatter["type"] == "kernel"
    assert search.get_page("sources/docs/triton-ascend-api.md").frontmatter["id"] == "doc-triton-ascend-api"
    with pytest.raises(KeyError, match="page not found"):
        search.get_page("missing")

    hits = search.grep(["pair_base"], scope="all", context=1)
    assert hits and hits[0].line_number > 0
    any_hits = search.grep(["RMSNorm", "Elementwise"], scope="wiki", any_match=True, context=0, limit=2)
    assert len({hit.path for hit in any_hits}) == 2
    artifact_hits = search.grep(["reference_rmsnorm_rope"], scope="artifacts", extensions=["py"])
    assert artifact_hits[0].path.endswith("rmsnorm_rope.py")
    with pytest.raises(ValueError, match="unsupported"):
        search.grep(["x"], scope="bad")


def test_page_type_detection() -> None:
    assert KernelWikiSearch.detect_page_type({"type": "kernel"}, "wiki/kernels/x.md") == "wiki-kernel"
    assert KernelWikiSearch.detect_page_type({}, "sources/prs/repo/PR-1.md") == "source-pr"
    assert KernelWikiSearch.detect_page_type({}, "other/x.md") == "unknown"
