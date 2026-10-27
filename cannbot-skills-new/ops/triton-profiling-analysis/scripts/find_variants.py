#!/usr/bin/env python3
# find_variants.py — 列出 triton 缓存中某 kernel 的所有编译变体，辅助辨认
# "哪个缓存目录对应哪个 shape/constexpr/编译选项"。
#
# 用法: find_variants.py <kernel名> [cache根目录]
# 示例: find_variants.py _attn_fwd
import json
import re
import sys
import time
from pathlib import Path

kernel = sys.argv[1]
cache_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".triton" / "cache"

rows = []
for d in sorted(cache_root.iterdir()):
    ttir = d / f"{kernel}.ttir"
    if not ttir.exists():
        continue
    txt = ttir.read_text(errors="replace")
    # constexpr 线索：arith.constant 里的 c<数字>_i32
    consts = sorted(set(re.findall(r"%c(\d+)_i32", txt)), key=int)
    # tensor 形状线索
    shapes = sorted(set(re.findall(r"tensor<(\d+x\d+x(?:bf16|f32|f16))>", txt)))
    # 循环数
    nfor = txt.count("scf.for")
    # 选项
    opts = {}
    j = d / f"{kernel}.json"
    if j.exists():
        try:
            meta = json.loads(j.read_text(errors="replace"))
            flat = json.dumps(meta)
            for key in ["num_stages", "multibuffer", "workspace_multibuffer"]:
                m = re.search(rf'"{key}":\s*(null|true|false|\d+)', flat)
                if m:
                    opts[key] = m.group(1)
        except Exception:
            pass
    mt = time.strftime("%m-%d %H:%M", time.localtime(j.stat().st_mtime if j.exists() else d.stat().st_mtime))
    rows.append((d.name, mt, nfor, consts[:8], shapes[:4], opts))

print(f"kernel={kernel}  共 {len(rows)} 个变体\n")
for name, mt, nfor, consts, shapes, opts in rows:
    print(f"{name}")
    print(f"  mtime={mt}  scf.for={nfor}  consts={consts}  shapes={shapes}")
    if opts:
        print(f"  opts={opts}")
    print()
print("提示: consts 里的大数通常对应 N_CTX/维度；结合编译时的 shape 与选项批次(mtime)定位目标变体。")
