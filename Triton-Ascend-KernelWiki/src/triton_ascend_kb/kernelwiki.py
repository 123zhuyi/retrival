from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable

from .io import MarkdownDocument, iter_markdown, load_yaml, read_markdown


TAG_FIELDS = (
    "tags",
    "techniques",
    "hardware_features",
    "kernel_types",
    "aliases",
    "symptoms",
    "operators",
)
CODE_EXTENSIONS = {
    ".cu",
    ".cuh",
    ".ptx",
    ".py",
    ".cpp",
    ".h",
    ".hpp",
    ".inl",
    ".txt",
    ".sh",
    ".yaml",
    ".json",
    ".yml",
    ".patch",
    ".c",
    ".cc",
    ".cxx",
    ".hxx",
    ".pyx",
}


@dataclass
class Page:
    path: str
    frontmatter: dict[str, Any]
    body: str
    page_type: str
    score: int = 0

    @property
    def id(self) -> str:
        return str(self.frontmatter.get("id", ""))

    @property
    def title(self) -> str:
        return str(self.frontmatter.get("title", "Untitled"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "type": self.page_type,
            "path": self.path,
            "score": self.score,
            "frontmatter": self.frontmatter,
        }


@dataclass(frozen=True)
class SearchFilters:
    type: str | None = None
    tag: str | None = None
    repo: str | None = None
    architecture: str | None = None
    symptom: str | None = None
    confidence: str | None = None
    has_code: bool = False


@dataclass(frozen=True)
class GrepHit:
    path: str
    line_number: int
    line: str
    context: tuple[str, ...] = field(default_factory=tuple)


class KernelWikiSearch:
    """Reproduces KernelWiki's filter, alias, and keyword-score behavior."""

    def __init__(self, root: Path):
        self.root = root
        self._aliases: dict[str, str] | None = None
        self._architecture_aliases: dict[str, list[str]] | None = None

    @staticmethod
    def _architecture_key(value: str) -> str:
        return re.sub(r"[\s_-]+", "", value.strip().lower())

    def load_architecture_expansions(self) -> dict[str, list[str]]:
        if self._architecture_aliases is not None:
            return self._architecture_aliases
        path = self.root / "data" / "architecture-aliases.yaml"
        raw = load_yaml(path) or {} if path.exists() else {}
        expansions: dict[str, list[str]] = {}
        for selector, models in raw.items():
            if not isinstance(selector, str):
                continue
            values = [models] if isinstance(models, str) else models or []
            expansions[self._architecture_key(selector)] = [
                str(model) for model in values if isinstance(model, str)
            ]
        self._architecture_aliases = expansions
        return expansions

    def expand_architecture(self, selector: str) -> list[str]:
        key = self._architecture_key(selector)
        expansions = self.load_architecture_expansions()
        # Direct: selector is a family selector (e.g. "Ascend910B", "A2", "910b").
        direct = expansions.get(key)
        if direct:
            return list(dict.fromkeys([selector, *direct]))
        # Reverse lookup: selector is a specific SKU (e.g. "Ascend910B3") that is not
        # itself a family key. Collect sibling SKUs from every family that contains
        # it, so querying a specific chip model still matches pages tagged with a
        # sibling model of the same family (e.g. B3 query matches a B1-tagged page).
        siblings: list[str] = []
        seen = {key}
        for family_skus in expansions.values():
            normalized_family = {self._architecture_key(sku) for sku in family_skus}
            if key in normalized_family:
                for sib in family_skus:
                    sib_key = self._architecture_key(sib)
                    if sib_key not in seen:
                        siblings.append(sib)
                        seen.add(sib_key)
        if siblings:
            return list(dict.fromkeys([selector, *siblings]))
        return [selector.strip()]

    def load_alias_expansions(self) -> dict[str, str]:
        if self._aliases is not None:
            return self._aliases
        path = self.root / "data" / "aliases.yaml"
        raw = load_yaml(path) or {} if path.exists() else {}
        aliases: dict[str, str] = {}
        for canonical, variants in raw.items():
            if not isinstance(canonical, str):
                continue
            aliases.setdefault(canonical.lower(), canonical)
            for variant in variants or []:
                if isinstance(variant, str):
                    aliases.setdefault(variant.lower(), canonical)
        self._aliases = aliases
        return aliases

    def expand_keyword(self, keyword: str) -> list[str]:
        architecture_values = self.expand_architecture(keyword)
        if len(architecture_values) > 1 or architecture_values[0].lower() != keyword.lower():
            return list(dict.fromkeys([keyword, *architecture_values]))
        canonical = self.load_alias_expansions().get(keyword.lower())
        if canonical and canonical.lower() != keyword.lower():
            return [keyword, canonical]
        return [keyword]

    @staticmethod
    def detect_page_type(frontmatter: dict[str, Any], path: str) -> str:
        if "type" in frontmatter:
            return f"wiki-{frontmatter['type']}"
        parts = path.replace("\\", "/").split("/")
        if parts[0] == "sources" and len(parts) > 1:
            return f"source-{parts[1].rstrip('s')}"
        return "unknown"

    def load_pages(self) -> list[Page]:
        pages: list[Page] = []
        for document in iter_markdown(self.root, ("sources", "wiki")):
            if not document.frontmatter:
                continue
            relative = document.path.relative_to(self.root).as_posix()
            pages.append(
                Page(
                    path=relative,
                    frontmatter=document.frontmatter,
                    body=document.body,
                    page_type=self.detect_page_type(document.frontmatter, relative),
                )
            )
        return pages

    def score_keyword_match(self, page: Page, keywords: Iterable[str]) -> int:
        title_text = str(page.frontmatter.get("title", "")).lower()
        tag_values: list[str] = []
        for key in TAG_FIELDS:
            value = page.frontmatter.get(key) or []
            if isinstance(value, str):
                value = [value]
            tag_values.extend(str(item) for item in value)
        tag_text = " ".join(tag_values).lower()
        body_lower = page.body.lower()
        score = 0
        for keyword in keywords:
            best_variant_score = 0
            for variant in self.expand_keyword(keyword):
                variant_lower = variant.lower()
                variant_score = 0
                if variant_lower in title_text:
                    variant_score += 10
                if variant_lower in tag_text:
                    variant_score += 5
                variant_score += min(body_lower.count(variant_lower), 3)
                best_variant_score = max(best_variant_score, variant_score)
            score += best_variant_score
        return score

    def _matches_filters(self, page: Page, filters: SearchFilters) -> bool:
        fm = page.frontmatter
        if filters.type and not page.page_type.endswith(filters.type) and page.page_type != filters.type:
            return False
        if filters.tag:
            tags: set[str] = set()
            for key in ("tags", "techniques", "hardware_features", "kernel_types"):
                value = fm.get(key) or []
                tags.update([value] if isinstance(value, str) else value)
            variants = {item.lower() for item in self.expand_keyword(filters.tag)}
            if not any(str(tag).lower() in variants for tag in tags):
                return False
        if filters.repo and filters.repo.lower() not in str(fm.get("repo", "")).lower():
            return False
        if filters.architecture:
            architectures = {str(value).lower() for value in fm.get("architectures") or []}
            variants = {item.lower() for item in self.expand_architecture(filters.architecture)}
            if not architectures.intersection(variants):
                return False
        if filters.symptom and filters.symptom not in set(fm.get("symptoms") or []):
            return False
        if filters.confidence and str(fm.get("confidence", "")) != filters.confidence:
            return False
        if filters.has_code and not self._page_has_code(page):
            return False
        return True

    def _page_has_code(self, page: Page) -> bool:
        candidates: list[Path] = []
        artifact_dir = page.frontmatter.get("artifact_dir")
        if artifact_dir:
            candidates.append(self.root / str(artifact_dir))
        for candidate in candidates:
            if candidate.is_dir() and any(
                item.is_file() and item.suffix.lower() in CODE_EXTENSIONS for item in candidate.rglob("*")
            ):
                return True
        return False

    def source_ids(self, frontmatter: dict[str, Any]) -> list[str]:
        """Return canonical source links, accepting the v0.2 legacy key."""
        value = frontmatter.get("sources")
        if value is None:
            value = frontmatter.get("source_pages")
        if value is None:
            return []
        return [str(value)] if isinstance(value, str) else [str(item) for item in value]

    def artifact_files(self, document: MarkdownDocument) -> list[Path]:
        artifact_dir = document.frontmatter.get("artifact_dir")
        if not artifact_dir:
            return []
        base = (self.root / str(artifact_dir)).resolve()
        try:
            base.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError(f"artifact_dir escapes wiki root: {artifact_dir}") from exc
        if not base.is_dir():
            return []
        return [
            path
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS
        ]

    def search(
        self,
        query: Iterable[str] = (),
        *,
        filters: SearchFilters | None = None,
        limit: int = 10,
    ) -> list[Page]:
        filters = filters or SearchFilters()
        keywords = [token for item in query for token in re.split(r"\s+", item.strip()) if token]
        pages = [page for page in self.load_pages() if self._matches_filters(page, filters)]
        if keywords:
            for page in pages:
                page.score = self.score_keyword_match(page, keywords)
            pages = [page for page in pages if page.score > 0]
            pages.sort(key=lambda page: (-page.score, page.path))
        else:
            pages.sort(key=lambda page: page.path)
        return pages[:limit]

    def get_page(self, page_id_or_path: str) -> MarkdownDocument:
        direct = self.root / page_id_or_path
        if direct.is_file():
            return read_markdown(direct)
        for page in self.load_pages():
            if page.id == page_id_or_path:
                return read_markdown(self.root / page.path)
        raise KeyError(f"page not found: {page_id_or_path}")

    def grep(
        self,
        patterns: Iterable[str],
        *,
        scope: str = "all",
        any_match: bool = False,
        context: int = 1,
        limit: int = 20,
        extensions: Iterable[str] = (),
    ) -> list[GrepHit]:
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        subdirs = {
            "wiki": ("wiki",),
            "sources": ("sources",),
            "all": ("wiki", "sources"),
            "artifacts": ("artifacts",),
        }.get(scope)
        if subdirs is None:
            raise ValueError(f"unsupported grep scope: {scope}")
        requested = {f".{item.strip().lstrip('.').lower()}" for item in extensions if item.strip()}
        if scope == "artifacts":
            allowed = CODE_EXTENSIONS | {".md"} | requested
        else:
            allowed = {".md"} | requested
        search_subdirs = list(subdirs)
        if requested and scope == "all":
            search_subdirs.append("artifacts")

        hits: list[GrepHit] = []
        matched_files = 0
        files: list[Path] = []
        for subdir in search_subdirs:
            base = self.root / subdir
            if base.exists():
                files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in allowed)
        for path in sorted(set(files)):
            relative = path.relative_to(self.root).as_posix()
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            file_hits: list[GrepHit] = []
            for index, line in enumerate(lines):
                matched = any(pattern.search(line) for pattern in compiled) if any_match else all(
                    pattern.search(line) for pattern in compiled
                )
                if not matched:
                    continue
                start, end = max(0, index - context), min(len(lines), index + context + 1)
                file_hits.append(
                    GrepHit(
                        path=relative,
                        line_number=index + 1,
                        line=line,
                        context=tuple(lines[start:end]),
                    )
                )
            if file_hits:
                hits.extend(file_hits)
                matched_files += 1
                if matched_files >= limit:
                    return hits
        return hits
