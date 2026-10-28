#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""last_pass.mlir 编译后 intrinsic 清单分析（片上层实现量/速率归因输入）。

方法论（2026-10-26 固化，用户指正驱动）：
- GM 层实现量用 ttir 解析（pipe_time_model.ir_traffic，形状清晰、循环内外可判）
- 片上层用 last_pass intrinsic call 清单——set_l1(UB→L1 预载)/fixpipe(L0C→UB 排流)/
  nd2nz(转置读通道)/vcast(数值重铸) 是编译器结构的唯一证据源，ttir 完全不可见

输出：逐 pipe 的 call 站点数、静态形状证据、（可选）与 instr_exe.csv 实测指令数
的交叉倍数（子 tile 化程度）。纯 stdlib，无 torch 依赖。
"""
import argparse
import collections
import csv
import glob
import json
import re

# intrinsic 名前缀 → (pipe, 语义)。顺序敏感（前缀匹配，先专后通）。
INTRINSIC_MAP = [
    ("load_gm_to_ubuf", "MTE2", "GM→UB 读"),
    ("nd2nz", "MTE2", "GM→UB 转置读（硬件转置通道）"),
    ("nz2nd", "MTE2", "GM→UB 转置读"),
    ("set_l1", "MTE1", "UB→L1 dot 操作数预载（编译器新增层）"),
    ("load_l1", "MTE1", "L1 相关"),
    ("fixpipe", "FIXP", "L0C→UB dot 输出排流（编译器新增层）"),
    ("store_ubuf_to_gm", "MTE3", "UB→GM 写"),
    ("mma_tile", "CUBE", "dot（_ta=转置A）"),
    ("mmad", "CUBE", "dot"),
    ("set_flag", "SYNC", "同步"),
    ("wait_flag", "SYNC", "同步"),
]

DT_SIZE = {"bf16": 2, "f16": 2, "fp16": 2, "f32": 4, "fp32": 4, "i32": 4, "i64": 8, "u32": 4, "i8": 1}
MEMREF_RE = re.compile(r"memref<([0-9x?]+)x?([a-z0-9]*)")
CALL_RE = re.compile(r"func\.call @([a-zA-Z_0-9]+)")


def classify(name):
    for pref, pipe, sem in INTRINSIC_MAP:
        if pref in name:
            return pipe, sem
    if name.startswith("v") or "broadcast_scalar" in name or name.startswith("vcast"):
        return "VECTOR", "向量算术/铸型"
    return "ADDR", "地址算术/控制（标量编排侧）"


def kernel_body(lines, kernel_hint=None):
    """截取目标 kernel 的 func 体；无 hint 时取 call 数最多的 func.func。"""
    funcs = []  # (name, start, end)
    cur = None
    for i, l in enumerate(lines):
        m = re.match(r"\s*func\.func (?:private )?@([a-zA-Z_0-9]+)\(", l)
        if m:
            cur = [m.group(1), i, None]
        elif cur and l.rstrip() == "}" and funcs and funcs[-1][2] is None and cur[1] != i:
            pass
        if m:
            funcs.append([m.group(1), i, None])
        elif funcs and funcs[-1][2] is None and l.rstrip() == "}":
            funcs[-1][2] = i
    if kernel_hint:
        for f in funcs:
            if f[0] == kernel_hint:
                return f
    best = max(funcs, key=lambda f: sum(1 for l in lines[f[1]:(f[2] or len(lines))] if "func.call" in l), default=None)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("last_pass", help="ir_dump/*_last_pass.mlir")
    ap.add_argument("--kernel-body", default=None, help="目标 func 名（缺省=call 最多的 func）")
    ap.add_argument("--instr-csv", default=None,
                    help="instr_exe.csv（可含通配符）：逐 pipe 实测指令数，与站点数交叉得子 tile 倍数")
    ap.add_argument("--json", default=None, help="结果落盘路径")
    a = ap.parse_args()

    lines = open(a.last_pass, errors="ignore").readlines()
    fn = kernel_body(lines, a.kernel_body)
    if not fn:
        print("ERROR: 未找到 func.func 体", file=__import__("sys").stderr)
        raise SystemExit(2)
    body = lines[fn[1]:(fn[2] or len(lines))]

    calls = collections.Counter(CALL_RE.search(l).group(1) for l in body if CALL_RE.search(l))
    # hivm.hir.set_flag/wait_flag 是 op 不是 func.call，单独计数（同步密度证据）
    for op in ("set_flag", "wait_flag"):
        calls[f"hivm.hir.{op}"] = sum(1 for l in body if f".{op}" in l)
    pipes = collections.defaultdict(lambda: {"sites": 0, "sem": "", "intrinsics": {}})
    for name, cnt in calls.items():
        pipe, sem = classify(name)
        pipes[pipe]["sites"] += cnt
        pipes[pipe]["sem"] = sem
        pipes[pipe]["intrinsics"][name] = cnt

    # 静态形状证据（核体内静态 memref，聚合去重；动态 ? 不计字节，仅列证据）
    shapes = collections.Counter()
    for l in body:
        for m in MEMREF_RE.finditer(l):
            dims, dt = m.group(1), m.group(2) or "?"
            shapes[f"{dims}:{dt}"] += 1
    shape_list = []
    for s, c in shapes.most_common(20):
        dims, dt = s.rsplit(":", 1)
        try:
            n = 1
            for d in dims.split("x"):
                n *= int(d)
            nb = n * DT_SIZE.get(dt, 0)
        except ValueError:
            nb = None
        shape_list.append({"shape_dtype": s, "count": c, "bytes_if_static": nb})

    # 同步密度
    sync = {k: calls.get(f"hivm.hir.{k}", 0) for k in ("set_flag", "wait_flag")}
    for k in ("hivm.hir.set_flag", "hivm.hir.wait_flag"):
        calls.pop(k, None)  # 不混入 pipe 站点计数

    instr = None
    if a.instr_csv:
        agg = collections.defaultdict(lambda: [0, 0])
        for f in glob.glob(a.instr_csv, recursive=True):
            for r in csv.DictReader(open(f, errors="ignore")):
                agg[r["pipe"]][0] += int(r["call_count"])
                agg[r["pipe"]][1] += int(r["cycles"])
        instr = {p: {"instr_calls": c, "cycles": cyc}
                 for p, (c, cyc) in sorted(agg.items(), key=lambda kv: -kv[1][1])}
        for p, inv in pipes.items():
            if p in instr and inv["sites"]:
                inv["instr_per_site"] = round(instr[p]["instr_calls"] / inv["sites"], 1)

    result = {
        "kernel_body": fn[0], "body_lines": [fn[1] + 1, (fn[2] or len(lines))],
        "total_call_sites": sum(calls.values()),
        "pipes": {p: v for p, v in sorted(pipes.items(), key=lambda kv: -kv[1]["sites"])},
        "sync": sync,
        "static_shape_evidence": shape_list,
        "note": "站点数=源层发射点；instr_per_site>1 即编译器子 tile 化倍数；"
                "MTE1/FIXP 站点为 ttir 不可见的编译器新增片上层",
    }
    if instr:
        result["instr_exe_by_pipe"] = instr
    out = json.dumps(result, ensure_ascii=False, indent=1)
    if a.json:
        open(a.json, "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
