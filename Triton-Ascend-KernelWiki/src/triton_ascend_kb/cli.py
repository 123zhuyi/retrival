from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .agent_bundle import KernelWikiAgentAdapter
from .indexer import build_indices, corpus_status, validate_corpus
from .io import dump_yaml
from .kernelwiki import KernelWikiSearch, SearchFilters


def _root(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"KernelWiki root does not exist: {path}")
    return path


def _add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--type")
    parser.add_argument("--tag")
    parser.add_argument("--repo")
    parser.add_argument("--architecture")
    parser.add_argument("--symptom")
    parser.add_argument("--confidence")
    parser.add_argument("--has-code", action="store_true")


def _filters(args: argparse.Namespace) -> SearchFilters:
    return SearchFilters(
        type=args.type,
        tag=args.tag,
        repo=args.repo,
        architecture=args.architecture,
        symptom=args.symptom,
        confidence=args.confidence,
        has_code=args.has_code,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triton-ascend-kernelwiki")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query = subparsers.add_parser("query", help="KernelWiki-compatible page query")
    query.add_argument("--root", type=_root, required=True)
    query.add_argument("query", nargs="*")
    _add_filters(query)
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--compact", action="store_true")
    query.add_argument("--paths-only", action="store_true")
    query.add_argument("--json", action="store_true")

    get_page = subparsers.add_parser("get-page", help="Read one page by id or path")
    get_page.add_argument("--root", type=_root, required=True)
    get_page.add_argument("page")
    view = get_page.add_mutually_exclusive_group()
    view.add_argument("--body-only", action="store_true")
    view.add_argument("--frontmatter-only", action="store_true")
    get_page.add_argument("--follow-sources", action="store_true")
    get_page.add_argument("--include-code", action="store_true")
    get_page.add_argument("--json", action="store_true")

    grep = subparsers.add_parser("grep", help="Regex search across wiki and sources")
    grep.add_argument("--root", type=_root, required=True)
    grep.add_argument("patterns", nargs="+")
    grep.add_argument("--scope", "--only", dest="scope", choices=["wiki", "sources", "all", "artifacts"], default="all")
    grep.add_argument("--any", action="store_true")
    grep.add_argument("--context", type=int, default=1)
    grep.add_argument("--limit", type=int, default=20)
    grep.add_argument("--files-only", action="store_true")
    grep.add_argument("--ext", help="Comma-separated extra extensions")
    grep.add_argument("--json", action="store_true")

    bundle = subparsers.add_parser("bundle", help="Build an Agent bundle from KernelWiki pages")
    bundle.add_argument("--root", type=_root, required=True)
    bundle.add_argument("query", nargs="+")
    _add_filters(bundle)
    bundle.add_argument("--limit", type=int, default=5)
    bundle.add_argument("--json", action="store_true")

    build_index = subparsers.add_parser("build-index", help="Generate KernelWiki query indices")
    build_index.add_argument("--root", type=_root, required=True)

    validate = subparsers.add_parser("validate", help="Validate KernelWiki pages and provenance links")
    validate.add_argument("--root", type=_root, required=True)

    status = subparsers.add_parser("repo-status", help="Print KernelWiki corpus counts")
    status.add_argument("--root", type=_root, required=True)
    return parser


def _run_query(args: argparse.Namespace) -> int:
    pages = KernelWikiSearch(args.root).search(args.query, filters=_filters(args), limit=args.limit)
    if args.json:
        print(json.dumps([page.to_dict() for page in pages], ensure_ascii=False, indent=2, default=str))
    elif args.paths_only:
        for page in pages:
            print(page.path)
    elif args.compact:
        for page in pages:
            print(f"  [{page.page_type}] {page.id}: {page.title}  ({page.path})")
    else:
        if not pages:
            print("No matching pages.")
        else:
            print(f"# {len(pages)} result(s)\n")
            for page in pages:
                print(f"## {page.title}")
                print(f"- **id**: `{page.id}`")
                print(f"- **type**: `{page.page_type}`")
                print(f"- **path**: `{page.path}`")
                for key in ("architectures", "confidence", "reproducibility", "tags", "techniques", "hardware_features", "kernel_types", "sources"):
                    value = page.frontmatter.get(key)
                    if value:
                        print(f"- **{key}**: {value}")
                print()
    return 0


def _run_get_page(args: argparse.Namespace) -> int:
    document = KernelWikiSearch(args.root).get_page(args.page)
    search = KernelWikiSearch(args.root)
    if args.json:
        print(
            json.dumps(
                {
                    "path": document.path.relative_to(args.root).as_posix(),
                    "frontmatter": document.frontmatter,
                    "body": document.body,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    elif args.frontmatter_only:
        print(dump_yaml(document.frontmatter).rstrip())
    elif args.body_only:
        print(document.body)
    else:
        print("---")
        print(dump_yaml(document.frontmatter).rstrip())
        print("---")
        print(document.body)

    if args.follow_sources:
        for source_id in search.source_ids(document.frontmatter):
            source = search.get_page(source_id)
            print(f"\n---\n## Source: {source_id}\n")
            print(source.body[:500].strip())

    if args.include_code:
        for path in search.artifact_files(document):
            relative = path.relative_to(args.root).as_posix()
            print(f"\n### `{relative}`\n")
            print(f"```{path.suffix.lstrip('.')}")
            print(path.read_text(encoding="utf-8", errors="replace")[: 200 * 1024])
            print("```")
    return 0


def _run_grep(args: argparse.Namespace) -> int:
    hits = KernelWikiSearch(args.root).grep(
        args.patterns,
        scope=args.scope,
        any_match=args.any,
        context=args.context,
        limit=args.limit,
        extensions=(args.ext or "").split(","),
    )
    if args.json:
        print(json.dumps([hit.__dict__ for hit in hits], ensure_ascii=False, indent=2, default=str))
    elif args.files_only:
        for path in dict.fromkeys(hit.path for hit in hits):
            print(path)
    else:
        for hit in hits:
            print(f"{hit.path}:{hit.line_number}: {hit.line}")
    return 0


def _run_bundle(args: argparse.Namespace) -> int:
    bundle = KernelWikiAgentAdapter(args.root).build(args.query, filters=_filters(args), limit=args.limit)
    if args.json:
        print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print(bundle.to_markdown())
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "query":
            return _run_query(args)
        if args.command == "get-page":
            return _run_get_page(args)
        if args.command == "grep":
            return _run_grep(args)
        if args.command == "bundle":
            return _run_bundle(args)
        if args.command == "build-index":
            for path in build_indices(args.root):
                print(path.relative_to(args.root).as_posix())
            return 0
        if args.command == "validate":
            errors = validate_corpus(args.root)
            if errors:
                for error in errors:
                    print(f"ERROR: {error}")
                return 1
            print("OK: KernelWiki validation passed")
            return 0
        if args.command == "repo-status":
            for line in corpus_status(args.root):
                print(line)
            return 0
    except (KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
