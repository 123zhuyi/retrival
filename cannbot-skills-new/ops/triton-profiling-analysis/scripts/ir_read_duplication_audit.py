#!/usr/bin/env python3
"""通用 IR 读点重复审计：任意 last_pass.mlir 函数节的重复读入证据生成器。

不预设算子、拆分数、编译器操作名。对节内每个 func.call 的每个 memref 操作数：
  追溯根入参 → 引用计数 → 循环位置 → 地址算式链展开，
并输出 pid/子核分解算式与 workspace 槽线索。结论（依赖集合→地址流数→重复次数）
由分析者按 channel-traffic-audit.md S4.5 五步判定，脚本只产证据。

用法:
  python3 ir_read_duplication_audit.py <last_pass.mlir> <节名> [选项]
    --ops nd2nz,load_gm_to_ubuf   只看名字含这些子串的调用（默认全部 func.call）
    --hints hints.json            可选: {"arg_names": {"%arg3": "k"}, "program_count": 16}

仅用标准库。
"""
import argparse
import json
import re
import sys


def split_funcs(text):
    parts = re.split(r'func\.func @([\w]+)', text)
    return {parts[i]: parts[i + 1].split("\n") for i in range(1, len(parts), 2)}


def build_defs(lines):
    defs = {}
    for j, l in enumerate(lines):
        m = re.match(r'\s+(%[\w]+)\s*=', l)
        if m:
            defs[m.group(1)] = j
    return defs


def find_loop(lines):
    for j, l in enumerate(lines):
        if re.search(r'\bscf\.for\b', l):
            ind = len(l) - len(l.lstrip())
            for k in range(j + 1, len(lines)):
                if lines[k].strip().startswith("}") and (len(lines[k]) - len(lines[k].lstrip())) <= ind:
                    return (j, k)
    return None


def parse_signature(line):
    """提取 memref 入参及其属性标记（workspace / tensor_kind=1 输出 / 普通）。"""
    out = []
    for m in re.finditer(r'(%arg\d+):\s*memref<[^{]+?\{([^}]*)\}', line):
        arg, attrs = m.group(1), m.group(2)
        kind = "普通"
        at = re.search(r'arg_type<(\w+)>', attrs)
        if at:
            kind = f"arg_type={at.group(1)}"
        elif re.search(r'tensor_kind\s*=\s*1', attrs):
            kind = "输出(tensor_kind=1)"
        out.append((arg, kind))
    return out


def trace_root(defs, lines, ssa, max_hops=40):
    """沿 def-use 链上溯到第一个 %argN。"""
    cur = ssa
    for _ in range(max_hops):
        if re.match(r'%arg\d+$', cur):
            return cur
        d = defs.get(cur)
        if d is None:
            return None
        refs = [r for r in re.findall(r'%[\w]+', lines[d]) if r != cur]
        hit = [r for r in refs if re.match(r'%arg\d+$', r)]
        if hit:
            return hit[0]
        if not refs:
            return None
        cur = refs[0]
    return None


def unwind(defs, lines, ssa, max_depth=10, breadth=6):
    """展开定义链，返回 (行号, 截断行) 列表（只取搬算相关操作）。"""
    kw = (r'reinterpret_cast|affine\.apply|memref\.subview|memref\.view|memref\.collapse_shape'
          r'|index_cast|arith\.addi|arith\.muli|arith\.divsi|arith\.remsi|memref\.load')
    seen, frontier, out = set(), [ssa], []
    for _ in range(max_depth):
        nxt = []
        for s in frontier:
            d = defs.get(s)
            if d is None:
                continue
            l = lines[d]
            if re.search(kw, l):
                out.append((d, l.strip()[:150]))
                for r in re.findall(r'%[\w]+', l):
                    if r != s and r not in seen:
                        seen.add(r)
                        nxt.append(r)
        frontier = nxt[:breadth]
        if not frontier:
            break
    return out


def decomp_lines(lines, window=80):
    """pid/子核分解算式：get_*_idx 之后窗口内的常量除法/取模/乘法。"""
    hits = []
    anchor = [j for j, l in enumerate(lines[:window]) if re.search(r'get_block_idx|get_sub_block_idx', l)]
    if not anchor:
        return hits
    lo = anchor[0]
    hi = max(j for j, l in enumerate(lines[:window]) if re.search(r'divsi|remsi', l)) if \
        any(re.search(r'divsi|remsi', l) for l in lines[lo:window]) else lo
    for j in range(lo, min(hi + 1, len(lines))):
        if re.search(r'arith\.(divsi|remsi|muli)', lines[j]):
            hits.append((j, lines[j].strip()[:130]))
    return hits


def map_bodies(lines):
    """全节扫描 affine_map 定义体。"""
    body = "\n".join(lines)
    out = {}
    for m in re.finditer(r'#map\d+\s*=\s*affine_map<\((.*?)\) -> \((.*?)\)>', body):
        out[f"#map[{m.group(1)}]->[{m.group(2)}]"] = m.group(0)[:120]
    return list(out.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir")
    ap.add_argument("section")
    ap.add_argument("--ops", default="", help="逗号分隔的调用名子串过滤（默认全部 func.call）")
    ap.add_argument("--hints", default="", help="JSON: arg_names / program_count")
    a = ap.parse_args()

    funcs = split_funcs(open(a.mlir).read())
    if a.section not in funcs:
        print(f"找不到节 {a.section}; 现有: {list(funcs)}")
        return 1
    lines = funcs[a.section]
    defs = build_defs(lines)
    loop = find_loop(lines)
    hints = {}
    if a.hints:
        hints = json.load(open(a.hints))
    names = hints.get("arg_names", {})
    nprog = hints.get("program_count")

    print(f"# 节 {a.section}  scf.for={loop}  program总数={nprog or '(分析者填)'}")

    # A. 签名 memref 参数
    print("\n## A. 入参表（来自签名属性，非猜测）")
    for arg, kind in parse_signature(lines[0]):
        print(f"  {arg}: {kind}  名称={names.get(arg, '?')}  全节引用={len([1 for l in lines if arg in l])} 处")

    # B. 分解算式
    print("\n## B. pid/子核分解算式（分析者标注各分量含义; 取模即拆分维度候选）")
    for j, l in decomp_lines(lines):
        print(f"  L{j}: {l}")

    # C. 调用点逐操作数审计
    ops_filter = [s for s in a.ops.split(",") if s]
    print("\n## C. 调用点逐 memref 操作数审计")
    for j, l in enumerate(lines):
        m = re.search(r'func\.call @(\w+)\s*\(([^)]*)\)', l)
        if not m:
            continue
        cname, cargs = m.group(1), [x.strip() for x in m.group(2).split(",")]
        if ops_filter and not any(s in cname for s in ops_filter):
            continue
        lp = "循环内" if (loop and loop[0] < j < loop[1]) else "循环外"
        print(f"\n  L{j} [{lp}] {cname}  操作数={cargs}")
        for idx, op in enumerate(cargs):
            if not re.match(r'%[\w]+', op):
                continue
            root = trace_root(defs, lines, op)
            if root is None:
                print(f"    op[{idx}] {op}: 根=? (非入参路径, 如 L0C/常量)")
                continue
            print(f"    op[{idx}] {op} → {root} 名称={names.get(root, '?')} 引用={len([1 for x in lines if root in x])}处")
            for d, ul in unwind(defs, lines, op):
                print(f"      L{d}: {ul}")

    # D. workspace 视图与 affine map
    print("\n## D. workspace 视图槽（偏移含 pid → 私有 → 交换非重复）")
    wargs = [arg for arg, kind in parse_signature(lines[0]) if kind == "arg_type=workspace"]
    for j, l in enumerate(lines):
        for wa in wargs:
            if re.search(r'memref\.view ' + re.escape(wa) + r'\[', l):
                print(f"  L{j}: {l.strip()[:130]}")
    print("\n## E. affine_map 定义体（地址算式的语义解码用）")
    for tag, raw in map_bodies(lines):
        print(f"  {raw}")

    print("\n# 通用判定规则（S4.5）:")
    print("#  地址依赖集合不含某拆分分量且该分量变量整节不存在 → 该维度 program 重复读;")
    print("#  重复次数 = program总数 / 地址流条数; workspace 槽偏移含 pid = 私有交换;")
    print("#  地址含循环变量 = 每迭代新数据; 采信门: IR倍数 vs 实测 <= 1.05")
    return 0


if __name__ == "__main__":
    sys.exit(main())
