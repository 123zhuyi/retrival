#!/usr/bin/env python3
"""IR #loc 锚点 → 源码行定位表（ttir / ttadapter / last_pass.mlir 通用）。

skill 3A.1「逐行 IR 形态核对」的脚本化：把每条 MLIR 操作的 #locN 解析回
源文件:行:列，并按标量降级形态归类（指针张量 gather / 逐元素算术 /
cumsum / 常量未标记），产出可直接贴进 §4.5 报告的定位表。

用法:
  ir_loc_resolver.py <kernel.mlir> [--op tt.load ...] [--gather-only] [--top N]

输出三段：
  1) loc 统计（#loc 定义数 / 含 callsite 链解析数）
  2) 逐 op 定位表：mlir 行 | op | #locN | 源文件:行:列 | 源码语句
  3) 标量降级形态归类汇总：各形态命中的 op 数 + 源码行清单（去重）
"""
import argparse, os, re
from collections import defaultdict

LOC_DEF = re.compile(r'^#loc(\d+)\s*=\s*loc\("([^"]+)":(\d+):(\d+)\)')
LOC_CALLSITE = re.compile(r'^#loc(\d+)\s*=\s*loc\(callsite\(#loc(\d+) at #loc(\d+)\)\)')
LOC_FUSED = re.compile(r'^#loc(\d+)\s*=\s*loc\(fused\[#loc(\d+)(?:, #loc\d+)*\]\)')
LOC_USE = re.compile(r'loc\(#loc(\d+)\)')
GATHER_TYPE = re.compile(r'tensor<\d+x!tt\.ptr')
SCALAR_ARITH = re.compile(r'arith\.(minsi|minf|maxsi|maxf|cmpf|cmpi|select|andi|ori|xori)\b')
FILE_LINES = {}


def load_locs(path):
    """解析 #loc 定义（含 callsite/fused 链）→ {id: (file, line, col)}。"""
    defs, chains = {}, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            m = LOC_DEF.match(raw)
            if m:
                defs[int(m.group(1))] = (m.group(2), int(m.group(3)), int(m.group(4)))
                continue
            c = LOC_CALLSITE.match(raw)
            if c:
                chains[int(c.group(1))] = int(c.group(2))
                continue
            fz = LOC_FUSED.match(raw)
            if fz:
                chains[int(fz.group(1))] = int(fz.group(2))
    resolved = dict(defs)
    for cid, tgt in chains.items():
        if tgt in defs:
            resolved[cid] = defs[tgt]
    return resolved, len(defs)


def src_stmt(path, line, maxlen=72):
    try:
        if path not in FILE_LINES:
            with open(path, encoding="utf-8", errors="replace") as f:
                FILE_LINES[path] = f.readlines()
        ls = FILE_LINES[path]
        if 1 <= line <= len(ls):
            s = ls[line - 1].strip()
            return s if len(s) <= maxlen else s[: maxlen - 1] + "…"
    except Exception:
        pass
    return ""


def op_name(code):
    m = re.search(r'=\s*(tt\.[\w.]+|scf\.[\w.]+|arith\.[\w.]+|func\.[\w.]+)', code)
    if m:
        return m.group(1)
    m = re.match(r'\s*(tt\.[\w.]+|scf\.[\w.]+|arith\.[\w.]+|cf\.[\w.]+|return)\b', code)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("--op", action="append", default=[], help="只看指定 op（可多次，如 --op tt.load --op scf.for）")
    ap.add_argument("--gather-only", action="store_true", help="只列指针张量 gather 相关行")
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args()

    locs, n_defs = load_locs(a.mlir)
    rows = []
    with open(a.mlir, encoding="utf-8", errors="replace") as f:
        for no, raw in enumerate(f, 1):
            m = LOC_USE.search(raw)
            if not m:
                continue
            lid = int(m.group(1))
            code = raw.rstrip()
            rows.append({
                "line": no, "lid": lid, "op": op_name(code.strip()),
                "gather": bool(GATHER_TYPE.search(raw)),
                "scalar_arith": bool(SCALAR_ARITH.search(raw)),
                "cumsum": "cumsum" in raw,
                "tgt": locs.get(lid), "code": code.strip()[:80],
            })

    print(f"## loc 统计：#loc 定义 {n_defs}，可解析锚点 {len(locs)}，引用行 {len(rows)}")
    all_rows = rows  # 形态归类基于全量行，不受表格过滤影响
    if a.op:
        rows = [r for r in rows if r["op"] in a.op]
    if a.gather_only:
        rows = [r for r in rows if r["gather"] and r["op"] in ("tt.load", "tt.store", "tt.splat", "tt.addptr")]

    print(f"\n## 逐 op 定位表（{min(len(rows), a.top)}/{len(rows)} 行）")
    print(f"{'mlir行':>7} | {'op':<22} | {'#loc':>5} | {'源文件:行:列':<28} | 源码语句")
    for r in rows[: a.top]:
        if r["tgt"]:
            f0, l0, c0 = r["tgt"]
            fl = f"{os.path.basename(f0)}:{l0}:{c0}"
            st = src_stmt(f0, l0)
        else:
            fl, st = "（未解析）", ""
        print(f"{r['line']:>7} | {r['op']:<22} | #{r['lid']:>4} | {fl:<28} | {st}")

    # ---- 形态归类汇总（skill 3A.1 第 2 步的脚本化）----
    def uniq_src(rows_):
        seen = {}
        for r in rows_:
            if r["tgt"]:
                f0, l0, _ = r["tgt"]
                seen[(os.path.basename(f0), l0)] = src_stmt(f0, l0)
        return seen

    cats = [
        ("指针张量 gather（tt.load/store tensor<Nx!tt.ptr>）",
         [r for r in all_rows if r["gather"] and r["op"] in ("tt.load", "tt.store")]),
        ("指针张量构造（splat/addptr tensor<Nx!tt.ptr>，gather 的前驱）",
         [r for r in all_rows if r["gather"] and r["op"] in ("tt.splat", "tt.addptr")]),
        ("逐元素 i32/f32 比较-算术（minsi/minf/cmp/select）", [r for r in all_rows if r["scalar_arith"]]),
        ("一维 cumsum / 标量归约", [r for r in all_rows if r["cumsum"]]),
    ]
    print("\n## 标量降级形态归类汇总")
    for title, rs in cats:
        srcs = uniq_src(rs)
        print(f"- {title}：{len(rs)} 个 op，命中源码行 {len(srcs)} 处")
        for (fb, ln), st in sorted(srcs.items(), key=lambda kv: kv[0][1])[:12]:
            print(f"    {fb}:{ln:<5} | {st}")


if __name__ == "__main__":
    main()
