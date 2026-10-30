from __future__ import annotations

from triton_ascend_kb.agent_bundle import KernelWikiAgentAdapter
from triton_ascend_kb.kernelwiki import SearchFilters


def test_bundle_contains_ranked_pages_and_followed_sources(wiki_root) -> None:
    bundle = KernelWikiAgentAdapter(wiki_root).build(
        ["rmsnorm", "vecutil_low"],
        filters=SearchFilters(type="kernel", architecture="Ascend910B1"),
        limit=3,
    )
    assert [page.id for page in bundle.pages] == ["kernel-rmsnorm-rope"]
    assert {source.id for source in bundle.source_excerpts} == {"pr-triton-ascend-fixture-993", "run-rmsnorm-rope-001"}
    result = bundle.to_dict()
    assert result["retrieval"] == "kernelwiki_title_tag_body"
    assert result["pages"][0]["body"].startswith("# RMSNorm")
    markdown = bundle.to_markdown()
    assert "KernelWiki Agent Knowledge Bundle" in markdown
    assert "Cited Source Excerpts" in markdown
    assert "final_rank" not in markdown


def test_bundle_deduplicates_sources_and_honors_excerpt_limit(wiki_root) -> None:
    adapter = KernelWikiAgentAdapter(wiki_root, source_excerpt_chars=30)
    bundle = adapter.build(["tail_mask"], limit=10)
    ids = [source.id for source in bundle.source_excerpts]
    assert len(ids) == len(set(ids))
    assert all(len(source.excerpt) <= 30 for source in bundle.source_excerpts)


def test_bundle_reports_unresolved_source(isolated_wiki) -> None:
    page = isolated_wiki / "wiki" / "patterns" / "broken-source.md"
    page.write_text(
        "---\nid: pattern-broken\ntitle: Broken source\ntype: pattern\ntags: [broken]\n"
        "confidence: inferred\nsources: [source-missing]\n---\nBody\n",
        encoding="utf-8",
    )
    bundle = KernelWikiAgentAdapter(isolated_wiki).build(["broken"])
    assert bundle.warnings == ["unresolved source page: source-missing"]
    assert "# Warnings" in bundle.to_markdown()
