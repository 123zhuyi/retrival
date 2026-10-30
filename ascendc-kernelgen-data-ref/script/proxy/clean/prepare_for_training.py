#!/usr/bin/env python3
"""第四步：清理目录，让 *.raw.json 直接可用于训练。

两件事：
  1. 删除目录下的 *.clean.json 和 *.diff.json（第三步的中间产物）
  2. 对剩余的 *.raw.json 删除顶层 session_id / meta / tools 字段，原地修改

用法：
    python3 scripts/prepare_for_training.py work/cleaned/
    python3 scripts/prepare_for_training.py work/cleaned/ --dry-run
    python3 scripts/prepare_for_training.py work/cleaned/ --fields session_id meta tools _meta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_FIELDS = ("session_id", "meta", "tools")
REMOVE_SUFFIXES = (".clean.json", ".diff.json")


def strip_sample(obj: Dict[str, Any], fields: List[str]) -> bool:
    """删除 obj 顶层指定的字段，返回是否删过任何字段。"""
    touched = False
    for f in fields:
        if f in obj:
            obj.pop(f)
            touched = True
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dir", help="待清理的目录")
    ap.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS),
                    help=f"要从 .raw.json 顶层删除的字段（默认 {' '.join(DEFAULT_FIELDS)}）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印会做什么，不实际修改")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        print(f"[err] {d} 不是目录", file=sys.stderr)
        return 2

    # 1. 删中间产物
    n_removed = 0
    for suf in REMOVE_SUFFIXES:
        for f in sorted(d.rglob(f"*{suf}")):
            if args.dry_run:
                print(f"[dry] rm {f.name}")
            else:
                f.unlink()
            n_removed += 1
    print(f"[prep] 删除中间文件 {n_removed} 个（*.clean.json / *.diff.json）")

    # 2. 删字段
    n_files = 0
    for f in sorted(d.rglob("*.raw.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[warn] {f.name} JSON 解析失败，跳过：{e}", file=sys.stderr)
            continue

        if isinstance(data, dict):
            touched = strip_sample(data, args.fields)
        elif isinstance(data, list):
            touched = False
            for s in data:
                if isinstance(s, dict):
                    touched = strip_sample(s, args.fields) or touched
        else:
            print(f"[warn] {f.name} 顶层类型 {type(data).__name__}，跳过", file=sys.stderr)
            continue

        if args.dry_run:
            print(f"[dry] strip {args.fields} from {f.name}")
        else:
            f.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        n_files += 1
    print(f"[prep] 处理 .raw.json 文件 {n_files} 个（删除字段：{args.fields}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
