from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .kernelwiki import KernelWikiSearch, SearchFilters


@dataclass(frozen=True)
class BundlePage:
    id: str
    title: str
    page_type: str
    path: str
    score: int
    frontmatter: dict[str, Any]
    body: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "type": self.page_type,
            "path": self.path,
            "score": self.score,
            "frontmatter": self.frontmatter,
            "body": self.body,
        }


@dataclass(frozen=True)
class SourceExcerpt:
    id: str
    title: str
    path: str
    excerpt: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "path": self.path, "excerpt": self.excerpt}


@dataclass
class AgentKnowledgeBundle:
    query: tuple[str, ...]
    pages: list[BundlePage]
    source_excerpts: list[SourceExcerpt]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": list(self.query),
            "retrieval": "kernelwiki_title_tag_body",
            "pages": [page.to_dict() for page in self.pages],
            "source_excerpts": [source.to_dict() for source in self.source_excerpts],
            "warnings": self.warnings,
        }

    def to_markdown(self) -> str:
        lines = ["# KernelWiki Agent Knowledge Bundle", "", f"Query: `{' '.join(self.query)}`", ""]
        for page in self.pages:
            lines.extend(
                [
                    f"## {page.title}",
                    "",
                    f"- Page: `{page.id}`",
                    f"- Type: `{page.page_type}`",
                    f"- Path: `{page.path}`",
                    f"- KernelWiki score: `{page.score}`",
                    "",
                    page.body.strip(),
                    "",
                ]
            )
        if self.source_excerpts:
            lines.extend(["# Cited Source Excerpts", ""])
            for source in self.source_excerpts:
                lines.extend(
                    [
                        f"## {source.title}",
                        "",
                        f"Source: `{source.id}` (`{source.path}`)",
                        "",
                        source.excerpt,
                        "",
                    ]
                )
        if self.warnings:
            lines.extend(["# Warnings", "", *[f"- {warning}" for warning in self.warnings], ""])
        return "\n".join(lines).rstrip() + "\n"


class KernelWikiAgentAdapter:
    """Thin adapter: query pages, fetch full bodies, and follow cited source pages."""

    def __init__(self, root: Path, *, source_excerpt_chars: int = 500):
        self.root = root
        self.search = KernelWikiSearch(root)
        self.source_excerpt_chars = source_excerpt_chars

    def build(
        self,
        query: Iterable[str],
        *,
        filters: SearchFilters | None = None,
        limit: int = 5,
    ) -> AgentKnowledgeBundle:
        query_tuple = tuple(query)
        ranked = self.search.search(query_tuple, filters=filters, limit=limit)
        pages: list[BundlePage] = []
        source_excerpts: list[SourceExcerpt] = []
        warnings: list[str] = []
        followed: set[str] = set()

        for page in ranked:
            document = self.search.get_page(page.path)
            pages.append(
                BundlePage(
                    id=page.id,
                    title=page.title,
                    page_type=page.page_type,
                    path=page.path,
                    score=page.score,
                    frontmatter=page.frontmatter,
                    body=document.body,
                )
            )
            for source_id in self.search.source_ids(page.frontmatter):
                source_id = str(source_id)
                if source_id in followed:
                    continue
                followed.add(source_id)
                try:
                    source = self.search.get_page(source_id)
                except KeyError:
                    warnings.append(f"unresolved source page: {source_id}")
                    continue
                source_excerpts.append(
                    SourceExcerpt(
                        id=source_id,
                        title=str(source.frontmatter.get("title", source_id)),
                        path=source.path.relative_to(self.root).as_posix(),
                        excerpt=source.body[: self.source_excerpt_chars].strip(),
                    )
                )
        return AgentKnowledgeBundle(query_tuple, pages, source_excerpts, warnings)
