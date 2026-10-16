#!/usr/bin/env python3
"""fix_ops_tags.py — 从 ops/ 算子卡的 frontmatter tags 里删除通用症状词.

这三个词在所有算子卡里几乎都出现, 导致跨算子误匹配.
"""
from __future__ import annotations

from pathlib import Path

import yaml

KB = Path("/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb").resolve()
OPS = KB / "ops"
REMOVE_TAGS = {"precision_mismatch", "ub_overflow", "performance_bottleneck"}


def _str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def _list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(str, _str_representer, Dumper=yaml.SafeDumper)
yaml.add_representer(list, _list_representer, Dumper=yaml.SafeDumper)


def fix_one(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    if end == -1:
        return False
    fm = yaml.safe_load(text[3:end]) or {}
    tags = fm.get("tags", [])
    if not isinstance(tags, list):
        return False
    new_tags = [t for t in tags if t not in REMOVE_TAGS]
    if len(new_tags) == len(tags):
        return False
    fm["tags"] = new_tags
    body = text[end + 4:]
    fm_text = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False, width=float("inf"))
    new_text = "---\n" + fm_text + "---" + body
    path.write_text(new_text, encoding="utf-8")
    return True


def main():
    changed = 0
    for md in sorted(OPS.glob("*.md")):
        if md.name == "index.md":
            continue
        if fix_one(md):
            print(f"fixed {md.name}: removed generic symptom tags")
            changed += 1
        else:
            print(f"skip  {md.name}")
    print(f"\n共修改 {changed} 张卡")


if __name__ == "__main__":
    main()
