#!/usr/bin/env python3
"""Import pinned Triton-Ascend product documentation into sources/docs/."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

import yaml

from _wiki_root import WIKI_ROOT
from _doc_tags import classify_document, load_rules


UPSTREAM_REPO = "triton-lang/triton-ascend"
UPSTREAM_URL = "https://github.com/triton-lang/triton-ascend"
GENERATED_BY = "scripts/ingest_triton_ascend_docs.py"
DOC_ROOTS = ("docs/en", "docs/zh")
DOC_EXTENSIONS = {".md", ".rst"}


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def verify_checkout(repo: Path, *, allow_unverified_origin: bool) -> None:
    origin = ""
    try:
        origin = git(repo, "config", "--get", "remote.origin.url")
    except subprocess.CalledProcessError:
        pass
    normalized = origin.lower().removesuffix(".git").replace(":", "/")
    expected = "github.com/triton-lang/triton-ascend"
    if expected not in normalized and not allow_unverified_origin:
        raise ValueError(f"checkout origin is not {UPSTREAM_REPO}: {origin or '<missing>'}")
    dirty = git(repo, "status", "--porcelain", "--", *DOC_ROOTS)
    if dirty:
        raise ValueError("documentation checkout has uncommitted changes; commit or discard them before ingestion")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def page_id(source_path: str) -> str:
    path = Path(source_path)
    without_suffix = path.with_suffix("").as_posix()
    return f"doc-triton-ascend-{slug(without_suffix.removeprefix('docs/'))}"


def title_from_document(path: Path, body: str) -> str:
    lines = body.splitlines()
    if path.suffix.lower() == ".md":
        for line in lines:
            match = re.match(r"^#\s+(.+?)\s*$", line)
            if match:
                return re.sub(r"[`*_]", "", match.group(1)).strip()
    for index, line in enumerate(lines[:-1]):
        if line.strip() and re.fullmatch(r"[=\-~^]+", lines[index + 1].strip()):
            return line.strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def topic_for(source_path: str) -> str:
    parts = Path(source_path).parts
    return slug(parts[2]) if len(parts) > 3 else "guide"


def render_page(source: Path, repo: Path, commit: str, retrieved_at: str, rules: dict) -> tuple[str, dict]:
    source_path = source.relative_to(repo).as_posix()
    body_bytes = source.read_bytes()
    body = body_bytes.decode("utf-8-sig")
    digest = hashlib.sha256(body_bytes).hexdigest()
    normalized_body = body.rstrip() + "\n"
    snapshot_digest = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
    language = source_path.split("/")[1]
    identifier = page_id(source_path)
    title = title_from_document(source, body)
    metadata = {
        "id": identifier,
        "title": title,
        "repo": UPSTREAM_REPO,
        "url": f"{UPSTREAM_URL}/blob/{commit}/{quote(source_path, safe='/')}",
        "source_category": "official-doc",
        "tags": ["triton_ascend", "official_doc", topic_for(source_path)],
        "retrieved_at": retrieved_at,
        "upstream_commit": commit,
        "upstream_path": source_path,
        "content_sha256": digest,
        "snapshot_body_sha256": snapshot_digest,
        "document_language": language,
        "source_format": source.suffix.lower().lstrip("."),
        "license": "MIT",
        "generated_by": GENERATED_BY,
    }
    for field, values in classify_document(
        upstream_path=source_path, title=title, body=body, rules=rules
    ).items():
        if values:
            metadata[field] = list(dict.fromkeys([*(metadata.get(field) or []), *values]))
    frontmatter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).rstrip()
    content = f"---\n{frontmatter}\n---\n\n{normalized_body}"
    relative_doc_path = source_path.removeprefix("docs/")
    local_doc_path = relative_doc_path if source.suffix.lower() == ".md" else f"{relative_doc_path}.md"
    record = {
        "id": identifier,
        "local_path": f"sources/docs/triton-ascend/{local_doc_path}",
        "upstream_path": source_path,
        "content_sha256": digest,
        "snapshot_body_sha256": snapshot_digest,
        "bytes": len(body_bytes),
    }
    return content, record


def discover(repo: Path) -> list[Path]:
    files: list[Path] = []
    for relative in DOC_ROOTS:
        base = repo / relative
        if not base.is_dir():
            raise FileNotFoundError(f"missing upstream documentation root: {relative}")
        files.extend(
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in DOC_EXTENSIONS and path.stat().st_size > 0
        )
    return sorted(files, key=lambda path: path.relative_to(repo).as_posix())


def write_if_changed(path: Path, content: str) -> bool:
    encoded = content.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return True


def ingest(repo: Path, *, retrieved_at: str, prune: bool) -> tuple[int, int, int]:
    commit = git(repo, "rev-parse", "HEAD")
    output_root = WIKI_ROOT / "sources" / "docs" / "triton-ascend"
    records = []
    changed = 0
    expected: set[Path] = set()
    seen_ids: set[str] = set()
    rules = load_rules()
    for source in discover(repo):
        content, record = render_page(source, repo, commit, retrieved_at, rules)
        if record["id"] in seen_ids:
            raise ValueError(f"generated duplicate page id: {record['id']}")
        seen_ids.add(record["id"])
        destination = WIKI_ROOT / record["local_path"]
        expected.add(destination.resolve())
        changed += int(write_if_changed(destination, content))
        records.append(record)

    removed = 0
    if prune and output_root.exists():
        for existing in output_root.rglob("*.md"):
            if existing.resolve() not in expected:
                existing.unlink()
                removed += 1

    manifest = {
        "upstream_repo": UPSTREAM_REPO,
        "origin_url": UPSTREAM_URL,
        "upstream_commit": commit,
        "retrieved_at": retrieved_at,
        "license": "MIT",
        "document_roots": list(DOC_ROOTS),
        "generated_by": GENERATED_BY,
        "page_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "pages": records,
    }
    manifest_text = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False)
    write_if_changed(WIKI_ROOT / "data" / "triton-ascend-doc-snapshot.yaml", manifest_text)
    return len(records), changed, removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Triton-Ascend docs into KernelWiki sources/docs")
    parser.add_argument("--repo-path", required=True, type=Path, help="Local checkout of triton-lang/triton-ascend")
    parser.add_argument("--retrieved-at", default=date.today().isoformat())
    parser.add_argument("--prune", action="store_true", help="Remove previously generated pages absent from this checkout")
    parser.add_argument("--allow-unverified-origin", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = args.repo_path.expanduser().resolve()
    if not (repo / ".git").exists():
        parser.error(f"not a git checkout: {repo}")
    verify_checkout(repo, allow_unverified_origin=args.allow_unverified_origin)
    count, changed, removed = ingest(repo, retrieved_at=args.retrieved_at, prune=args.prune)
    print(f"Imported {count} document pages; changed={changed}; removed={removed}")
    print(f"Pinned commit: {git(repo, 'rev-parse', 'HEAD')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
