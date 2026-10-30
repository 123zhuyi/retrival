#!/usr/bin/env python3
"""Reapply controlled symptom/kernel/operator tags without changing doc bodies."""

from __future__ import annotations

import yaml

from _doc_tags import classify_document, load_rules
from _wiki_root import WIKI_ROOT


GENERATED_BY = "scripts/ingest_triton_ascend_docs.py"
MANAGED_FIELDS = ("architectures", "hardware_features", "symptoms", "kernel_types", "operators")


def retag() -> tuple[int, int]:
    rules = load_rules()
    scanned = 0
    changed = 0
    root = WIKI_ROOT / "sources" / "docs" / "triton-ascend"
    for path in sorted(root.rglob("*.md")):
        raw = path.read_bytes()
        head, marker, body_bytes = raw.partition(b"\n---\n\n")
        if not marker or not head.startswith(b"---\n"):
            raise ValueError(f"invalid generated source page: {path.relative_to(WIKI_ROOT)}")
        frontmatter = yaml.safe_load(head[4:].decode("utf-8")) or {}
        if frontmatter.get("generated_by") != GENERATED_BY:
            continue
        scanned += 1
        body = body_bytes.decode("utf-8")
        classified = classify_document(
            upstream_path=str(frontmatter.get("upstream_path", "")),
            title=str(frontmatter.get("title", "")),
            body=body,
            rules=rules,
        )
        for field in MANAGED_FIELDS:
            values = classified[field]
            if values:
                frontmatter[field] = values
            else:
                frontmatter.pop(field, None)
        new_head = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).encode("utf-8").rstrip()
        updated = b"---\n" + new_head + marker + body_bytes
        if updated != raw:
            path.write_bytes(updated)
            changed += 1
    return scanned, changed


if __name__ == "__main__":
    scanned, changed = retag()
    print(f"Retagged {scanned} document pages; changed={changed}")
