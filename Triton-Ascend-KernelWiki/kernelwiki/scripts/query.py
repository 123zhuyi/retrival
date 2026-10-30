#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from _common import query_pages


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the Triton-Ascend KernelWiki")
    parser.add_argument("query", nargs="*")
    for name in ("type", "tag", "repo", "architecture", "symptom", "confidence"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--has-code", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--paths-only", action="store_true")
    args = parser.parse_args()
    keywords = [token for item in args.query for token in re.split(r"\s+", item.strip()) if token]
    result = query_pages(keywords, args)
    if args.paths_only:
        for page in result:
            print(page["path"])
        return 0
    if not result:
        print("No matching pages.")
        return 0
    print(f"# {len(result)} result(s)\n")
    for page in result:
        fm = page["fm"]
        if args.compact:
            print(f"  [{page['type']}] {fm.get('id', '')}: {fm.get('title', 'Untitled')}  ({page['path']})")
            continue
        print(f"## {fm.get('title', 'Untitled')}")
        print(f"- **id**: `{fm.get('id', '')}`")
        print(f"- **type**: `{page['type']}`")
        print(f"- **path**: `{page['path']}`")
        for key in ("architectures", "confidence", "reproducibility", "tags", "techniques", "hardware_features", "kernel_types", "sources"):
            if fm.get(key):
                print(f"- **{key}**: {fm[key]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
