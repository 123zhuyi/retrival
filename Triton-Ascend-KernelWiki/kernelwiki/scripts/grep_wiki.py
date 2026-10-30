#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from _common import CODE_EXTENSIONS, WIKI_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Text search across Triton-Ascend KernelWiki")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--only", choices=["wiki", "sources", "all", "artifacts"], default="all")
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--any", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--files-only", action="store_true")
    parser.add_argument("--ext")
    args = parser.parse_args()
    try:
        patterns = [re.compile(value, re.IGNORECASE) for value in args.patterns]
    except re.error as exc:
        parser.error(f"invalid regex: {exc}")
    dirs = {"wiki": ["wiki"], "sources": ["sources"], "all": ["wiki", "sources"], "artifacts": ["artifacts"]}[args.only]
    extra = {f".{value.strip().lstrip('.').lower()}" for value in (args.ext or "").split(",") if value.strip()}
    if extra and args.only == "all":
        dirs.append("artifacts")
    extensions = (CODE_EXTENSIONS | {".md"} | extra) if args.only == "artifacts" else ({".md"} | extra)
    matched = []
    for subdir in dirs:
        for path in sorted((WIKI_ROOT / subdir).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            hits = []
            for index, line in enumerate(lines):
                ok = any(p.search(line) for p in patterns) if args.any else all(p.search(line) for p in patterns)
                if ok:
                    start, end = max(0, index - args.context), min(len(lines), index + args.context + 1)
                    hits.append((index + 1, lines[start:end]))
            if hits:
                matched.append((path, hits))
    matched = matched[: args.limit]
    if args.files_only:
        for path, _ in matched:
            print(path.relative_to(WIKI_ROOT).as_posix())
        return 0
    if not matched:
        print("No matches.")
        return 0
    print(f"# {len(matched)} file(s) match")
    for path, hits in matched:
        print(f"\n## {path.relative_to(WIKI_ROOT).as_posix()}  ({len(hits)} matches)")
        for line_number, context in hits[:5]:
            print("```")
            for offset, line in enumerate(context):
                print(f"{line_number - min(args.context, line_number - 1) + offset}: {line}")
            print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
