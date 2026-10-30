#!/usr/bin/env python3
from __future__ import annotations

import argparse
import yaml

from _common import WIKI_ROOT, artifact_files, find_page, source_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve one Triton-Ascend KernelWiki page")
    parser.add_argument("lookup")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--body-only", action="store_true")
    group.add_argument("--frontmatter-only", action="store_true")
    parser.add_argument("--follow-sources", action="store_true")
    parser.add_argument("--include-code", action="store_true")
    args = parser.parse_args()
    _, fm, body = find_page(args.lookup)
    if args.frontmatter_only:
        print(yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).rstrip())
    elif args.body_only:
        print(body)
    else:
        print("---")
        print(yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).rstrip())
        print("---")
        print(body)
    if args.follow_sources:
        for source_id in source_ids(fm):
            _, source_fm, source_body = find_page(source_id)
            print(f"\n---\n## Source: {source_fm.get('title', source_id)} (`{source_id}`)\n")
            print(source_body[:500].strip())
    if args.include_code:
        for file in artifact_files(fm):
            print(f"\n### `{file.relative_to(WIKI_ROOT).as_posix()}`\n")
            print(f"```{file.suffix.lstrip('.')}")
            print(file.read_text(encoding="utf-8", errors="replace")[: 200 * 1024])
            print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
