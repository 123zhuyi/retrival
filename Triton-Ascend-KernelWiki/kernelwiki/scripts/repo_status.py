#!/usr/bin/env python3
from collections import Counter
import yaml

from _common import WIKI_ROOT, pages


if __name__ == "__main__":
    corpus = pages()
    types = Counter(page["type"] for page in corpus)
    modes = Counter()
    for path in (WIKI_ROOT / "artifacts").rglob("PROVENANCE.yaml"):
        modes[(yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("asset_mode", "unknown")] += 1
    print("Corpus counts:")
    print(f"  Indexed source/wiki pages: {len(corpus)}")
    print(f"  Source PR pages: {types['source-pr']}")
    print(f"  Wiki synthesis pages: {sum(count for kind, count in types.items() if kind.startswith('wiki-'))}")
    print(f"  Doc summaries: {types['source-doc']}")
    print(f"  Evaluation run pages: {types['source-run']}")
    print(f"  Candidate ledgers: {sum(1 for _ in (WIKI_ROOT / 'candidates').glob('*.yaml'))}")
    print(f"  Artifact bundles: {sum(modes.values())}")
    print(f"  Query indices: {sum(1 for _ in (WIKI_ROOT / 'queries').glob('*.md'))}")
    if modes:
        print("\nArtifact bundles by mode:")
        for mode, count in sorted(modes.items()):
            print(f"  {mode}: {count}")
