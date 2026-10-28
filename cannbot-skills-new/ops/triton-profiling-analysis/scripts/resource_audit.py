#!/usr/bin/env python3
"""双资源审计脚本（搬运×计算，量×速度）：一套采集 → 四象限证据 + 报告骨架。

通用性约定：
  - 只依赖 CSV 原始列 + last_pass.mlir 通用结构（func.call/memref/affine），不预设算子
  - MMA 计算量换算用 FLOP = 2×sqrt(A×B×C)（三个操作数 cast 元素数），无分形布局歧义，零算子知识
  - 峰值/带宽分母自动调用同级 chip_spec.py（探测 SOC）；失败则标注"分母缺失"，禁止编造
  - 变量命名、语义理论量、冗余"为什么"属算子知识 → 输出【分析者填写】占位，脚本不下结论

用法:
  python3 resource_audit.py --csv-dir <含5张CSV的OPPROF目录> [--mlir <last_pass.mlir>]
      [--hints hints.json] [--out report.md] [--json out.json]
  hints.json 可选字段: semantic_read_mb/semantic_write_mb（语义理论量）,
      arg_names {...}, nt(迭代数), programs(program数), field_unit_kb(Memory datas 单位,默认KB)

仅标准库。
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LINE_BYTES = 128  # 缓存行


def load_csv(d, name):
    p = os.path.join(d, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def fnum(rows, col, pred=None):
    vs = [float(r[col]) for r in rows if r.get(col) not in (None, "", "NA")]
    return (st.mean(vs), sum(vs)) if vs else (None, None)


def rows_by(rows):
    cube = [r for r in rows if r.get("sub_block_id") == "cube0"]
    vec = [r for r in rows if (r.get("sub_block_id") or "").startswith("vector")]
    return cube, vec


def split_funcs(text):
    parts = re.split(r'func\.func @([\w]+)', text)
    return {parts[i]: parts[i + 1].split("\n") for i in range(1, len(parts), 2)}


def mm_sites(lines):
    """MMA 调用点：返回 (行号, [三个操作数 cast 行]) —— 向上最近 6 行内的 cbuf/cc cast。"""
    out = []
    pat = re.compile(r'func\.call @(\w*mma\w*)\s*\(')
    cast = re.compile(r'memref\.cast\s+%(\w+)\s*:\s*memref<([\dx?]+)x(f32|bf16|f16)?[^>]*address_space<(cbuf|cc)>')
    for j, l in enumerate(lines):
        m = pat.search(l)
        if not m:
            continue
        shapes = []
        for k in range(max(0, j - 8), j):
            cm = cast.search(lines[k])
            if cm:
                dim = re.findall(r'\d+', cm.group(2))
                shapes.append((cm.group(4), eval_safe(dim)))
        if len(shapes) >= 3:
            out.append((j, m.group(1), shapes[-3:]))
    return out


def eval_safe(dims):
    n = 1
    for d in dims:
        n *= int(d)
    return n


def transfer_sites(lines):
    """搬运调用点几何（与逐变量归账同口径：gm 侧行走 + min 侧缓冲下界）。"""
    out = []
    pat = re.compile(r'func\.call @(\w*(?:nd2nz|nz2nd|load_gm_to_ubuf|store_ubuf_to_gm|fixpipe)\w*)\s*\(([^)]*)\)')
    defs = {}
    for j, l in enumerate(lines):
        m = re.match(r'\s+(%[\w]+)\s*=', l)
        if m:
            defs[m.group(1)] = j
    for j, l in enumerate(lines):
        m = pat.search(l)
        if not m:
            continue
        name, args = m.group(1), [x.strip() for x in m.group(2).split(",")]
        gi = 0 if ("load" in name or "nd2nz" in name) else 1
        if gi >= len(args):
            continue
        shape = walk_shape(defs, lines, args[gi], "gm")
        other = args[1 - gi] if len(args) > 1 else None
        shape_ob = walk_shape(defs, lines, other, "ub") if other else None
        job, row = (shape if shape else (None, None))
        if shape and shape_ob and shape_ob[0] < shape[0]:
            job = shape_ob[0]
        out.append((j, name, (job, row) if job else None))
    return out


def prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def trace_arg(defs, lines, ssa, hops=40):
    """沿 def 链上溯第一个 %argN（根入参）。"""
    cur = ssa
    for _ in range(hops):
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



def walk_shape(defs, lines, ssa, space, hops=12):
    """沿 def 链找 space 侧静态形状，返回 (字节数, 行连续段字节) 或 None。"""
    dt_of = lambda l: 2.0 if re.search(r'bf16|f16', l) else (4.0 if re.search(r'f32|float', l) else 2.0)
    cur = ssa
    for _ in range(hops):
        d = defs.get(cur)
        if d is None:
            return None
        dl = lines[d]
        sz = re.search(r'sizes:\s*\[([\d?, ]+)\]', dl)
        if sz and "reinterpret_cast" in dl:
            dims = [int(x) for x in re.findall(r'\d+', sz.group(1))]
            if not dims:
                return None
            return (prod(dims) * dt_of(dl), (dims[-2] if len(dims) >= 2 else prod(dims)) * dt_of(dl))
        def dims_of(body):
            b = re.sub(r'bf16|f16|f32|i32|i8|float|strided.*', '', body)
            ds = [int(x) for x in re.findall(r'\d+', b)]
            if ds and "?" not in b:
                return (prod(ds) * dt_of(dl), (ds[-2] if len(ds) >= 2 else prod(ds)) * dt_of(dl))
            return None
        if re.search(r'memref\.(cast|view)\s', dl) and f"address_space<{space}>" in dl:
            parts = dl.split(" to ", 1)
            if len(parts) == 2:
                got = dims_of(parts[1])  # 仅 TO 侧（源侧是行距/别名，不可信）
                if got:
                    return got
        if f"address_space<{space}>" in dl and "pointer_cast" in dl:
            sm = re.search(r'memref<([^>]+)>', dl)
            if sm:
                got = dims_of(sm.group(1))
                if got:
                    return got
        refs = [r for r in re.findall(r'%[\w]+', dl) if r != cur]
        if not refs:
            return None
        cur = refs[0]
    return None


def site_traffic(lines, nt, ncore, names=None):
    """逐站点流量归账（机械证据）：每调用点 = job字节 × nt × 核数，按根入参聚合。"""
    pat = re.compile(r'func\.call @(\w*(?:nd2nz|nz2nd|load_gm_to_ubuf|store_ubuf_to_gm|fixpipe)\w*)\s*\(([^)]*)\)')
    defs = {}
    for j, l in enumerate(lines):
        m = re.match(r'\s+(%[\w]+)\s*=', l)
        if m:
            defs[m.group(1)] = j
    dt_of = lambda l: 2.0 if re.search(r'bf16|f16', l) else (4.0 if re.search(r'f32|float', l) else 2.0)
    acc, unresolved = {}, 0
    for j, l in enumerate(lines):
        m = pat.search(l)
        if not m:
            continue
        name, args = m.group(1), [x.strip() for x in m.group(2).split(",")]
        gi = 0 if ("load" in name or "nd2nz" in name) else 1
        if gi >= len(args):
            continue
        shape = walk_shape(defs, lines, args[gi], "gm")
        other = args[1 - gi] if len(args) > 1 else None
        shape_ob = walk_shape(defs, lines, other, "ub") if other else None
        job, row = (shape if shape else (None, None))
        if shape and shape_ob and shape_ob[0] < shape[0]:
            job = shape_ob[0]  # 量取 min 侧缓冲下界（子核切分安全）
        # 行连续段恒取 gm 侧布局（min 侧行距不代表 gm 访存连续性）
        if not job:
            unresolved += 1
            continue
        root = trace_arg(defs, lines, args[gi]) or "?未解析"
        key = (root, (names or {}).get(root, ""), name.split("_")[0])
        acc.setdefault(key, []).append(job)
    rows = [(k[0], k[1], k[2], sum(v)/1e6*nt*ncore, st.mean(v)/1024, len(v))
            for k, v in sorted(acc.items(), key=lambda kv: -sum(kv[1]))]
    return rows, unresolved


def vec_issue_density(lines, nt, freq_hz, busy_us):
    """发射密度机械估算：aiv 内 v* 调用附近 ub 静态元素数 × nt ÷ 128 lane ÷ 忙时周期。"""
    pat = re.compile(r'func\.call @(v\w+)\s*\(([^)]*)\)')
    defs = {}
    for j, l in enumerate(lines):
        m = re.match(r'\s+(%[\w]+)\s*=', l)
        if m:
            defs[m.group(1)] = j
    total_el = 0
    for j, l in enumerate(lines):
        m = pat.search(l)
        if not m:
            continue
        els = [walk_shape(defs, lines, x.strip(), "ub") for x in m.group(2).split(",")]
        els = [e[0] for e in els if e]
        if els:
            total_el += min(els)  # 每调用点按最小侧元素数计一次
    busy_cyc = busy_us * 1e-6 * freq_hz
    issue_cyc = total_el * nt / 128.0
    return total_el, issue_cyc, (issue_cyc / busy_cyc * 100 if busy_cyc else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--kernel", required=True, help="kernel 名（必要产物标识，写入报告头与 manifest 字段）")
    ap.add_argument("--case", required=True, help="case 标识（如 c1；必要产物标识）")
    ap.add_argument("--mlir", default="")
    ap.add_argument("--hints", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--emit-sketch-draft", default="",
                    help="输出 kernel_sketch_draft.json（IR 重建路径：sketch.txt 缺失时供 agent 补全表达式后过机械闸）")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    hints = json.load(open(a.hints)) if a.hints else {}
    R = {"warnings": []}

    # ---- 阶段0 前置检查 ----
    opb = load_csv(a.csv_dir, "OpBasicInfo.csv")
    if not opb:
        print("FAIL: OpBasicInfo.csv 缺失")
        return 1
    if len(opb) != 1:
        R["warnings"].append(f"OpBasicInfo {len(opb)} 行（多发射），时长/占比口径需拆分")
    mem = load_csv(a.csv_dir, "Memory.csv")
    pu = load_csv(a.csv_dir, "PipeUtilization.csv")
    ar = load_csv(a.csv_dir, "ArithmeticUtilization.csv")
    l2 = load_csv(a.csv_dir, "L2Cache.csv")
    missing = [n for n, d in [("Memory", mem), ("PipeUtilization", pu), ("ArithmeticUtilization", ar)] if not d]
    if missing:
        print(f"FAIL: CSV 缺失 {missing}")
        return 1
    cube, vec = rows_by(mem)
    pcube, pvec = rows_by(pu)
    acube, avec = rows_by(ar)
    unit = hints.get("field_unit_mb", 0.001)  # datas 字段→MB 乘子（默认 KB 字段 ×0.001）

    # ---- 阶段1 峰值/带宽分母（chip_spec）----
    spec = {}
    try:
        out = subprocess.run([sys.executable, os.path.join(HERE, "chip_spec.py"), "--json"],
                             capture_output=True, text=True, timeout=30)
        spec = json.loads(out.stdout)
    except Exception as e:
        R["warnings"].append(f"chip_spec 不可用({e})——速度分母缺失，效率比本节禁报")
    peak_core_tf = spec.get("cube_peak_tflops_fp16_bf16_dense")
    if peak_core_tf:
        peak_core_tf = peak_core_tf / max(1, spec.get("cube_core_cnt", 1))
    bw = spec.get("pipe_bandwidth_gbs", {})
    l2r = (bw.get("L2读") or {}).get("per_core_gbs")
    l2w = (bw.get("L2写") or {}).get("per_core_gbs")

    # ---- 阶段2 搬运量（Memory.csv 原始计数）----
    def col_sum(rows, pat):
        tot = 0.0
        for r in rows:
            for c, v in r.items():
                if c and re.search(pat, c) and v not in ("", "NA"):
                    try:
                        tot += float(v)
                    except ValueError:
                        pass
        return tot
    mv = {}
    mv["GM→L1"] = col_sum(cube, r"GM_to_L1_datas") * unit
    mv["GM→UB"] = col_sum(vec, r"GM_to_UB_datas") * unit
    mv["L0C→GM"] = col_sum(cube, r"L0C_to_GM_datas") * unit
    mv["UB→GM"] = col_sum(vec, r"UB_to_GM_datas") * unit
    mv["读合计"] = mv["GM→L1"] + mv["GM→UB"]
    mv["写合计"] = mv["L0C→GM"] + mv["UB→GM"]
    mv["总搬运"] = mv["读合计"] + mv["写合计"]
    unit_kb = unit
    R["movement_mb"] = mv

    # ---- 阶段3 计算量（指令计数 × IR 分形换算）----
    instr_mean, instr_sum = fnum(acube, "aic_cube_fp_instr_number")
    flop_mma = None
    frag_note = ""
    if a.mlir and os.path.exists(a.mlir):
        text = open(a.mlir).read()
        funcs = split_funcs(text)
        sec = next((k for k in funcs if "mix_aic" in k), None) or next(iter(funcs))
        lines = funcs[sec]
        sites = mm_sites(lines)
        flop_per_tile = []
        for j, name, shapes in sites:
            el = [s[1] for s in shapes]
            if len(el) == 3 and all(el):
                flop_per_tile.append(2 * (el[0] * el[1] * el[2]) ** 0.5)
        if flop_per_tile and instr_mean:
            calls_per_core = instr_mean
            per_iter_est = len(sites)
            nt = hints.get("nt")
            if nt and abs(calls_per_core / nt - per_iter_est) > 0.51:
                R["warnings"].append(f"指令数 {calls_per_core:.0f}/核 ≠ IR 调用点 {per_iter_est}×NT{nt} —— 计算冗余或迭代口径不符")
            frag = flop_per_tile[0] / 8192.0
            flop_mma = instr_sum * frag * 8192
            frag_note = f"{len(sites)} 调用点/迭代 × 片数 {frag:.0f} × 8192"
        vecf_sum = fnum(avec, "aiv_vec_fops")[1]
    else:
        vecf_sum = None
        R["warnings"].append("未提供 --mlir，MMA 换算/归因检查跳过")
    R["flop_mma"] = flop_mma

    # ---- 阶段4 速度对账（忙时 vs 分母）----
    def busy(col, rows):
        return fnum(rows, col)[0] if rows else None
    speed = []
    cb = busy("aic_cube_time(us)", pcube)
    if cb and flop_mma and peak_core_tf:
        ncore = max(1, len(acube))
        rate = (flop_mma / ncore) / (cb * 1e-6) / 1e12
        occ = fnum(pcube, "aic_cube_ratio")[0] * 100
        be = rate / peak_core_tf * 100
        speed.append(("cube MMA(计算)", f"{rate:.2f} TFLOPS", f"{peak_core_tf:.2f}", be, occ, "aic_cube_ratio"))
    elif not peak_core_tf:
        R["warnings"].append("计算速度行缺分母，未输出")
    ch2spec = {"GM→L1": ("aic_mte2", l2r, pcube), "L0C→GM": ("aic_fixpipe", l2w, pcube),
               "GM→UB": ("aiv_mte2", l2r, pvec), "UB→GM": ("aiv_mte3", l2w, pvec)}
    for ch, (pcol, spkw, prows) in ch2spec.items():
        if not spkw:
            continue
        btime = fnum(prows, f"{pcol}_time(us)")[0]
        if not btime:
            continue
        ncore = len(prows) or 1
        rate = (mv[ch] / ncore) / (btime * 1e-6) / 1e3
        occ_c = [c for c in (prows[0] if prows else {}) if re.search(re.escape(pcol) + r".*ratio", c)]
        occ = fnum(prows, occ_c[0])[0] * 100 if occ_c else None
        occ_col = occ_c[0] if occ_c else ""
        be = rate / spkw * 100
        speed.append((ch, f"{rate:.1f} GB/s", f"{spkw:.1f}", be, occ, occ_col))
    R["speed"] = speed

    # ---- 阶段5 归因检查（跳读/块大小；lane 填充/发射密度）----
    attr = {"jobs": [], "note": []}
    if a.mlir and os.path.exists(a.mlir):
        funcs = split_funcs(open(a.mlir).read())
        for sec, lines in funcs.items():
            side = "cube" if "aic" in sec else "vec"
            for j, name, geo in transfer_sites(lines):
                if geo:
                    job_b, row_b = geo
                    attr["jobs"].append((side, name, row_b, job_b))
        jump = [j for (s, n, rb, jb) in attr["jobs"] if rb is not None and rb < LINE_BYTES]
        cb_jobs = [jb for (s, n, rb, jb) in attr["jobs"] if s == "cube"]
        vb_jobs = [jb for (s, n, rb, jb) in attr["jobs"] if s == "vec"]
        attr["jump_read"] = jump
        attr["cube_job_mean_kb"] = st.mean(cb_jobs) / 1024 if cb_jobs else None
        attr["vec_job_mean_kb"] = st.mean(vb_jobs) / 1024 if vb_jobs else None
        if vecf_sum and avec:
            vbusy = fnum(pvec, "aiv_vec_time(us)")[0]
            vcore = max(1, len(avec))
            if vbusy:
                freq = spec.get("cube_freq_mhz", 1800) * 1e6
    R["attribution"] = {k: v for k, v in attr.items()}

    # ---- 阶段5b 逐变量流量归账 + 发射密度（全部机械证据）----
    per_var = None
    if a.mlir and os.path.exists(a.mlir):
        funcs = split_funcs(open(a.mlir).read())
        names = hints.get("arg_names", {})
        ncore = max(1, len(acube))
        nvec = max(1, len(avec))
        mm_n = len(mm_sites(funcs[next(k for k in funcs if "mix_aic" in k)])) if any("mix_aic" in k for k in funcs) else 0
        nt = hints.get("nt") or (round(instr_mean / mm_n) if (instr_mean and mm_n) else None)
        if nt:
            R["nt_inferred"] = {"nt": nt, "来源": "hint" if hints.get("nt") else f"指令数{instr_mean:.0f}/调用点{mm_n}"}
            aic_sec = next((k for k in funcs if "mix_aic" in k), None)
            aiv_sec = next((k for k in funcs if "mix_aiv" in k), None)
            per_var = {}
            if aic_sec:
                r, un = site_traffic(funcs[aic_sec], nt, ncore, names)
                per_var["aic(GM→L1/L0C→GM)"] = (r, un)
            if aiv_sec:
                r, un = site_traffic(funcs[aiv_sec], nt, nvec, names)
                per_var["aiv(GM→UB/UB→GM)"] = (r, un)
            vbusy = fnum(pvec, "aiv_vec_time(us)")[0]
            if vbusy and aiv_sec:
                el, icy, den = vec_issue_density(funcs[aiv_sec], nt, spec.get("cube_freq_mhz", 1800) * 1e6, vbusy)
                if den:
                    R["vec_density"] = {"元素/迭代/核": el, "发射周期估算": round(icy), "忙时周期": round(vbusy * 1e-6 * spec.get("cube_freq_mhz", 1800) * 1e6), "发射密度%": round(den, 1)}
                else:
                    R["warnings"].append("vec 发射密度：v* 调用无可解析静态 ub 形状，悬置")

    # ---- 阶段5c 草图草稿（IR 重建路径）----
    if a.emit_sketch_draft and a.mlir and os.path.exists(a.mlir) and flop_mma and instr_mean:
        funcs = split_funcs(open(a.mlir).read())
        sec = next((k for k in funcs if "mix_aic" in k), next(iter(funcs)))
        sites = mm_sites(funcs[sec])
        nt = hints.get("nt") or (round(instr_mean / len(sites)) if sites else None)
        draft = {
            "provenance": "ir_reconstructed",  # 无 sketch.txt：设计意图不可考，锚 IR 实现事实
            "kernel": a.kernel, "case": a.case,
            "ir_refs": {"mlir": a.mlir, "mma_sites": [f"L{j+1} {n}" for j, n, _ in sites]},
            "constants": {"NT": nt, "programs": max(1, len(acube)),
                          "nvec": max(1, len(avec)),
                          "mma_tile_elems": [s[1] for s in sites[0][2]] if sites else None},
            "kernels": {sec: {"exprs_draft": {
                "c_flop": [f"site L{j+1} {n}: A={sh[0][1]} B={sh[1][1]} C={sh[2][1]} elems → 每 tile 2·√(A·B·C) FLOP × NT × programs（sqrt 不在白名单，agent 换整数乘积形）"
                           for j, n, sh in sites],
                "gm_in": "【agent 补】任务下限 + 复制因子（ir_read_duplication_audit 依赖集合→流数）",
                "gm_out": "【agent 补】语义必需输出（S2 契约）",
                "fixp_bytes": "【agent 补】L0C→GM 输出宽度",
                "v_flop": "【agent 补】v* 调用清单（归因检查发射密度段可数）"}},
                "measured_anchors": {"flop_mma": flop_mma, "channels_mb": mv,
                                     "closure_pct": {k: v for k, v in R.get("closure", {}).items()}}},
            "redundancy_notes": "【agent 补】对照任务下限逐 pipe 说明",
            "gates_note": "表达式须过白名单（数字/符号/四则/cdiv）+ gm_in≥任务下限 + md5 指纹，草稿未过闸，agent 补全后交 pipe_time_model --sketch",
        }
        json.dump(draft, open(a.emit_sketch_draft, "w"), ensure_ascii=False, indent=1)
        print(f"草图草稿（IR 重建） → {a.emit_sketch_draft}")

    # ---- 阶段6 输出 ----
    lines_out = []
    A = lines_out.append
    R["kernel"] = a.kernel
    R["case"] = a.case
    A(f"# 双资源审计报告（resource_audit.py 自动产出骨架）—— kernel: {a.kernel} / case: {a.case}")
    A("")
    A(f"采集目录: `{a.csv_dir}`｜发射行数: {len(opb)}")
    A("")
    A("## 【量·搬运】")
    A("| 通道 | 实测 MB |")
    A("|---|---|")
    for k in ["GM→L1", "GM→UB", "L0C→GM", "UB→GM", "读合计", "写合计", "总搬运"]:
        A(f"| {k} | {mv[k]:.2f} |")
    A("")
    A("**语义理论量（分析者填写）**: 读 ___ / 写 ___（hint semantic_read_mb/semantic_write_mb 可自动填）")
    sr, sw = hints.get("semantic_read_mb"), hints.get("semantic_write_mb")
    if sr and sw:
        A(f"→ 由 hints 填入: 读 {sr} / 写 {sw} = {sr+sw} MB；膨胀 {mv['总搬运']/(sr+sw):.2f}×")
    A("")
    A("## 【量·计算】")
    if flop_mma:
        A(f"- 执行指令数 {instr_mean:.0f}/核 × {len(acube)} 核 = {instr_sum:.0f}；MMA 换算 FLOP = 指令 × {frag_note} = **{flop_mma/1e6:.1f} MFLOP**")
        A("- 语义理论 FLOP（分析者按 2×M×N×K×迭代×程序 填写）: ___ → 膨胀 ___")
    else:
        A("- MMA 换算跳过（缺 mlir 或形状含 ?）")
    if vecf_sum:
        A(f"- 向量侧 aiv_vec_fops Σ = {vecf_sum/1e6:.1f} MFLOP（口径可信度高于 cube fops 列）")
    A("")
    A("## 【速度·MFU 分解】整体效率 = 占用占比 × 忙时效率")
    A("| pipe | 忙时速率 | 峰值/带宽 | 忙时效率 | 占用占比 | 整体效率 |")
    A("|---|---|---|---|---|---|")
    for name, rate, pk, be, occ, occ_col in speed:
        os_ = f"{occ:.1f}%" if occ else "—"
        A(f"| {name} | {rate} | {pk} | {be:.1f}% | {os_} | {be*occ/100:.2f}% |" if occ else
          f"| {name} | {rate} | {pk} | {be:.1f}% | — | — |")
    A("")
    A("## 【归因检查（机械部分）】")
    if attr.get("jobs"):
        A("| 行 | op | 行连续段B | 作业KB | 跳读? |")
        A("|---|---|---|---|---|")
        for side, name, rb, jb in attr["jobs"]:
            A(f"| {side} | {name[:38]} | {f'{rb:.0f}' if rb else '—'} | {jb/1024:.1f} | {'⚠️<缓存行' if (rb is not None and rb < LINE_BYTES) else '否'} |")
        A("")
        A("- 判读: 跳读列全部=否 → 跳读排除；作业KB 侧间差 → 块大小差异(小侧=descriptor开销占比高)")
        A("- 注: 几何=gm 侧 TO 静态形状（reinterpret_cast/view to/pointer_cast），量取 min 侧缓冲下界；异常行回 IR 人工核对")
    if per_var:
        A("")
        A("### 逐变量流量归账（机械：job字节 × NT × 核数，按根入参聚合）")
        side_ch = {"aic(GM→L1/L0C→GM)": {"nd2nz": "GM→L1", "fixpipe": "L0C→GM"},
                   "aiv(GM→UB/UB→GM)": {"load": "GM→UB", "store": "UB→GM"}}
        closure = {}
        for sec, (rows, un) in per_var.items():
            for root, nm, opc, mb, kb, n in rows:
                ch = side_ch.get(sec, {}).get(opc)
                if ch:
                    closure[ch] = closure.get(ch, 0) + mb
        R["closure"] = {ch: round(closure[ch] / mv[ch] * 100) for ch in closure if mv.get(ch)}
        for sec, (rows, un) in per_var.items():
            A(f"**{sec}**（未解析站点 {un}）")
            A("| 根入参 | 名称 | op类 | 流量MB | 单次KB | 站点数 |")
            A("|---|---|---|---|---|---|")
            for root, nm, opc, mb, kb, n in rows:
                A(f"| {root} | {nm or '【agent 定名】'} | {opc} | {mb:.2f} | {kb:.1f} | {n} |")
        A("- 逐侧闭合: " + "；".join(f"{ch} 归账{closure[ch]:.2f} vs 实测{mv.get(ch,0):.2f} ({closure[ch]/mv[ch]*100:.0f}%)" for ch in closure if mv.get(ch)))
        A("- 分析者: 未闭合侧回 IR 人工核对（子核切分/别名 cast）→ 命名 → 冗余分类（S4.5）")
    if R.get("vec_density"):
        d = R["vec_density"]
        A("")
        A(f"- vec 发射密度（机械估算）: {d['元素/迭代/核']} 元素/迭代/核 → 发射 {d['发射周期估算']} 周期 / 忙时 {d['忙时周期']} 周期 = **{d['发射密度%']}%**（其余=指令间空泡；lane 填充判定归 agent）")
    if R.get("nt_inferred"):
        A(f"- NT 推断: {R['nt_inferred']['nt']}（{R['nt_inferred']['来源']}）")
    A("")
    A("## 【分析者填写】变量命名 / 冗余为什么 / 草图反馈")
    A("（脚本只产证据；命名排除法、依赖集合→流数、A3 分类、草图回写见 references/channel-traffic-audit.md）")
    if R["warnings"]:
        A("")
        A("## 警告与悬置")
        for w in R["warnings"]:
            A(f"- {w}")
    report = "\n".join(lines_out)
    if a.out:
        open(a.out, "w").write(report)
        print(f"报告骨架 → {a.out}")
    if a.json:
        json.dump(R, open(a.json, "w"), ensure_ascii=False, indent=1, default=str)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
