#!/usr/bin/env python3
"""双流水源码变量统计：统计源代码中的变量是否形成双流水。

硬判据（用户定义）：某变量对应的缓冲拿到 **≥2 个互不相同的地址** 才算双流水成立——
  - 双参数 pointer_cast(offA, offB) 且 offA != offB：编译器原生乒乓对（最强证据）；
  - 跨多个 pointer_cast op 出现 ≥2 个不同偏移：多实例地址（次强）；
  - 只有 1 个地址、或多个 op 地址全相同：**不算**（两个地址必须不同）。

输入：bishengir --mlir-print-debuginfo 的全量 dump（含 pointer_cast + loc 源码行）。
      run_full_analysis 的产物 <kernel>_last_pass_mb.mlir.full 即此格式（multibuffer=True 口径）。

用法:
  dual_buffer_vars.py <dump1.txt> [dump2.txt ...] [--src <impl.py>] [--json out.json] [--quiet]
  # --src 提供源码文件时，行号解析为变量名（取该行 `var = ...` 的左值）；缺省用 file:line 标签
"""
import argparse, json, os, re, sys
from collections import defaultdict

PC_DUAL = re.compile(
    r'pointer_cast\(%c(\d+)_i64,\s*%c(\d+)_i64\)\s*:\s*memref<([^,>]+),\s*'
    r'#hivm\.address_space<(\w+)>>\s*loc\("([^"]+)":(\d+):(\d+)\)')
PC_SINGLE = re.compile(
    r'pointer_cast\(%c(\d+)_i64\)\s*:\s*memref<([^,>]+),\s*'
    r'#hivm\.address_space<(\w+)>>\s*loc\("([^"]+)":(\d+):(\d+)\)')

LHS = re.compile(r'^\s*(?:%[0-9]+\s*=\s*)?([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)')

_src_lines_cache = {}


def src_var(path, line):
    """源码行 → 左值变量名（无赋值语句则 None）。"""
    key = (path, line)
    if key not in _src_lines_cache:
        try:
            if path not in _src_lines_cache.get("__files__", {}):
                with open(path, encoding="utf-8", errors="replace") as f:
                    _src_lines_cache.setdefault("__files__", {})[path] = f.readlines()
            ls = _src_lines_cache["__files__"][path]
            _src_lines_cache[key] = ls[line - 1] if 1 <= line <= len(ls) else ""
        except Exception:
            _src_lines_cache[key] = ""
    m = LHS.match(_src_lines_cache[key] or "")
    return m.group(1) if m else None


def parse_dump(path):
    """dump → {kernel, vars: {(file,line): {var, shape, space, ops, dual_pairs,
               same_addr_duals, offsets}}}（按 kernel 从文件名或 module 名取）。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    ctx_counts = {}
    # 本 kernel 已有 multi_buffer 标记的空间分布（识别枝5：某空间被溢出回退整组关闭）
    import re as _re2, collections as _cl
    mark_spaces = set(_re2.findall(
        r'annotation\.mark \%\d+ \{hivm\.multi_buffer = \d+ : i32\} : memref<[^,>]+,\s*#hivm\.address_space<(\w+)>>', txt))
    kn = os.path.basename(path)
    m = re.search(r'_gdnv_\w+_kernel', kn) or re.search(r'@(_gdnv_\w+_kernel)', txt[:5000])
    kernel = m.group(0) if m else kn
    vars_ = defaultdict(lambda: {"ops": 0, "dual_pairs": 0, "same_addr_duals": 0,
                                 "offsets": set(), "shape": None, "space": None,
                                 "inplace_ctx": 0, "copy_ctx": 0})
    # locN → (file,line) 映射，用于统计每源码行的 IR 形态（判 mark 决策树挂枝）
    loc2line = {}
    for m2 in re.finditer(r'#loc(\d+) = loc\("([^"]+)":(\d+):\d+\)', txt):
        loc2line[int(m2.group(1))] = (m2.group(2), int(m2.group(3)))
    for ol in txt.splitlines():
        m2 = re.search(r'loc\(#[a-z]*(\d+)\)', ol)
        if not m2:
            continue
        key = loc2line.get(int(m2.group(1)))
        if not key:
            continue
        if "materialize_in_destination" in ol:
            ctx_counts.setdefault(key, {}).setdefault("inplace", 0)
            ctx_counts[key]["inplace"] += 1
        elif "memref.copy" in ol or "ND2NZ" in ol or "hivm.hir.load" in ol:
            ctx_counts.setdefault(key, {}).setdefault("copy", 0)
            ctx_counts[key]["copy"] += 1
    for a, b, shape, space, f, ln, col in PC_DUAL.findall(txt):
        v = vars_[(f, int(ln))]
        c = ctx_counts.get((f, int(ln)), {})
        if c.get("inplace"): v["inplace_ctx"] = 1
        if c.get("copy"): v["copy_ctx"] = 1
        v["ops"] += 1
        v["offsets"] |= {int(a), int(b)}
        v["shape"], v["space"] = shape, space
        if int(a) != int(b):
            v["dual_pairs"] += 1
        else:  # 硬判据防护：双参数同址不算双流水
            v["same_addr_duals"] += 1
    for off, shape, space, f, ln, col in PC_SINGLE.findall(txt):
        v = vars_[(f, int(ln))]
        v["ops"] += 1
        v["offsets"].add(int(off))
        if v["shape"] is None:
            v["shape"], v["space"] = shape, space
        c = ctx_counts.get((f, int(ln)), {})
        if c.get("inplace"): v["inplace_ctx"] = 1
        if c.get("copy"): v["copy_ctx"] = 1
    return kernel, vars_, mark_spaces


def judge(v):
    """硬判据：≥2 个互不相同地址。"""
    n = len(v["offsets"])
    if v["dual_pairs"] > 0:
        return "双流水成立（双缓冲对）"
    if n >= 2:
        return "双流水成立（跨op多副本）"
    if v["same_addr_duals"] > 0 and n == 1:
        return "不成立（双参数同址，两地址必须不同）"
    return "不成立（单副本）"


def buf_bytes(shape):
    """memref 形状 → 字节数（f32/i32=4，bf16/f16=2）。"""
    try:
        parts = shape.split("x")
        n = 1
        for d in parts[:-1]:
            n *= int(d)
        dt = parts[-1]
        return n * (2 if ("bf16" in dt or "f16" in dt) else 4)
    except Exception:
        return None


def unmarked_reason(v, mark_spaces=()):
    """未双缓冲原因（据 MarkMultiBuffer.cpp 决策树 + IR 形态 + 标记空间分布）：
    返回 (原因, 是否应双未双)。大且热的搬运/物化缓冲 → 应双未双（重点）。"""
    size = buf_bytes(v.get("shape") or "")
    space = v.get("space") or ""
    # 枝5识别：本 kernel 有标记但全落其它空间 → 本空间被溢出回退整组关闭
    gate5 = mark_spaces and space not in mark_spaces
    if v.get("inplace_ctx"):
        hot = v["ops"] >= 3 and (size or 0) >= 8192
        if gate5 and hot:
            return (f"UB/L1 溢出回退整组关闭（{space} 组无任何标记，源码 disableMultiBufferOnXX）", True)
        return (f"原地物化（根非独立 alloc，mark 枝2 排除；{(size or 0)//1024}KB×{v['ops']}次）", hot)
    if v.get("copy_ctx"):
        if (size or 0) >= 8192 and v["ops"] >= 5:
            if gate5:
                return (f"{space} 空间多缓冲被溢出回退整组关闭（本 kernel 标记全落 {mark_spaces}）", True)
            return (f"搬运目的地未获标（{(size or 0)//1024}KB 热缓冲，疑循环类型限制）", True)
        return (f"小搬运缓冲（{size}B，多为循环外不变量，乒乓收益可忽略）", False)
    if size is not None and size <= 4096:
        return (f"小缓冲（{size}B），乒乓收益可忽略", False)
    if v["ops"] <= 2:
        return ("低频访问", False)
    return ("计算型/标量值（无搬运 op 挂钩，mark 枝1 不适用）", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="全量 dump（--mlir-print-debuginfo）文件")
    ap.add_argument("--src", default=None, help="源码 .py（行号→变量名）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    out = {}
    for path in a.dumps:
        kernel, vars_, mark_spaces = parse_dump(path)
        rows = []
        for (f, ln), v in sorted(vars_.items(), key=lambda kv: -(kv[1]["dual_pairs"] or len(kv[1]["offsets"]))):
            name = src_var(a.src or f, ln) if (a.src or os.path.exists(f)) else None
            offs = sorted(v["offsets"])
            verdict = judge(v)
            reason, should = (unmarked_reason(v, mark_spaces) if verdict.startswith("不成立") else ("", False))
            rows.append({
                "var": name, "line": ln, "src": f"{os.path.basename(f or a.src or '?')}:{ln}",
                "shape": v["shape"], "space": v["space"], "ops": v["ops"],
                "size_bytes": buf_bytes(v.get("shape") or ""),
                "distinct_addrs": len(offs), "dual_pairs": v["dual_pairs"],
                "same_addr_duals": v["same_addr_duals"],
                "delta_bytes": ([offs[1] - offs[0]] if len(offs) == 2 else None),
                "offsets": offs[:8],
                "verdict": verdict, "reason": reason, "should_dual": should,
            })
        ok = [r for r in rows if r["verdict"].startswith("双流水成立")]
        out[kernel] = {"vars_total": len(rows), "dual_ok": len(ok),
                       "mark_spaces": sorted(mark_spaces), "vars": rows}
        if not a.quiet:
            print(f"===== {kernel}: {len(ok)}/{len(rows)} 个源码行变量双流水成立 =====")
            print(f"{'变量':<10} {'行':>5} {'空间':<5} {'副本数':>4} {'双缓冲对':>5} {'间距B':>8}  判定")
            for r in rows:
                d = r["delta_bytes"][0] if r["delta_bytes"] else ("—" if r["dual_pairs"] else "")
                print(f"{(r['var'] or '?'):<10} L{r['line']:<4} {r['space'] or '?':<5} "
                      f"{r['distinct_addrs']:>4} {r['dual_pairs']:>4} {str(d):>8}  {r['verdict']}")
            print()
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)
        if not a.quiet: print(f"[json] {a.json}")


if __name__ == "__main__":
    main()
