#!/usr/bin/env python3
"""双流水（double buffering / 软件流水）生效性判定：IR × trace 交叉。

IR 侧口径（依据 /path/AscendNPU-IR/docs，BISHENG/HIVM 权威文档）分两层：
  1. frontend_request（意图）：ttir/ttadapter 中 num_stages>1 循环属性——
     源码 tl.range(num_stages=N) 的**前端请求**，≠ 已落地（实测 triton-ascend 3.2
     的 .ttadapter 仅保留该属性，多缓冲不在此层发生）；
  2. backend_applied（落地）：CVPipelining/PlanMemory 的落地标记——
     {cube_loop}/{vector_loop} 循环属性、set-workspace-multibuffer/
     alloc_workspace、local_alloc/memdesc 多缓冲、insert/extract_slice 乒乓。
     triton-ascend 3.2 的 .ttadapter 实测全部缺失（下游 CANN 编译器完成），
     该层命中即证实；缺失不能证伪（需 trace 侧定实效）。

trace 侧（运行时实效，本 stack 上唯一可靠的"生效"证据）：
  a. 在途并发：pipe 原始忙碌和/归并占用 > 1 ⇒ 同 pipe 多笔在途（传输自身多缓冲）；
  b. 掩盖对重叠度（活跃区间=剔 WAIT_FLAG*/BAR，重叠度=交集/min(活跃)）：
     cube 侧 C_MTE2_MTE1 / C_MTE1_MAC(MTE1×CUBE) / C_MTE2_MAC；
     vec  侧 V_MTE2_VEC(MTE2×VECTOR) / V_VEC_MTE3(VECTOR×MTE3)。

判定（主对 = cube 侧 max(C_MTE1_MAC, C_MTE2_MAC)，vec 侧 V_MTE2_VEC）：
  重叠度 ≥30% → 生效；10–30% → 部分；<10%（掩盖失效线）→ 未生效。

用法:
  dual_pipeline_check.py --dir <OPPROF_*/simulator> [--ir-dir <.triton_cache>] \
      [--kernel _gdnv_state_kernel] [--json out.json]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

WAIT_PREFIX = "WAIT_FLAG"
CANONICAL = {  # 掩盖对 → (pipeA, pipeB, 说明)
    "C_MTE2_MTE1": ("MTE2", "MTE1", "GM搬入×L0装载"),
    "C_MTE1_MAC": ("MTE1", "CUBE", "L0装载×Cube计算"),
    "C_MTE2_MAC": ("MTE2", "CUBE", "GM搬入×Cube计算"),
    "V_MTE2_VEC": ("MTE2", "VECTOR", "GM搬入×Vector计算"),
    "V_VEC_MTE3": ("VECTOR", "MTE3", "Vector计算×搬出"),
}


def is_wait(name):
    return isinstance(name, str) and (name.startswith(WAIT_PREFIX) or name == "BAR")


def load_pipes(trace):
    """→ {pipe: {"raw": Σdur, "act": 活跃区间（剔等待）}}，按 process_name 元数据解析 pipe。"""
    d = json.load(open(trace))
    pid2name = {str(e["pid"]): e["args"]["name"] for e in d["traceEvents"]
                if e.get("ph") == "M" and e.get("name") == "process_name" and e.get("args", {}).get("name")}
    pipes = defaultdict(lambda: {"raw": 0.0, "act": []})
    for e in d["traceEvents"]:
        if e.get("ph") != "X": continue
        pipe = pid2name.get(str(e.get("pid")), str(e.get("tid", "?")))
        dur = float(e.get("dur") or 0)
        p = pipes[pipe]
        p["raw"] += dur
        if not is_wait(e.get("name")):
            p["act"].append((e["ts"], e["ts"] + dur))
    return pipes


def union(ivs):
    ivs = sorted(ivs)
    out = []
    for a, b in ivs:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total(ivs):
    return sum(b - a for a, b in ivs)


def inter(A, B):
    i = j = 0; s = 0.0
    while i < len(A) and j < len(B):
        a1, a2 = A[i]; b1, b2 = B[j]
        lo, hi = max(a1, b1), min(a2, b2)
        if hi > lo: s += hi - lo
        if a2 < b2: i += 1
        else: j += 1
    return s


def trace_side(simdir):
    """core0 子核（cube/vec 按存在自适应）→ 双流水指标。"""
    res = {"in_flight": {}, "pairs": {}}
    files = sorted(glob.glob(os.path.join(simdir, "core0.*", "trace.json")))
    if not files: return None
    for f in files:
        sub = os.path.basename(os.path.dirname(f))
        pipes = load_pipes(f)
        # 在途并发（传输 pipe）
        for pn in ("MTE2", "MTE1"):
            p = pipes.get(pn)
            if not p or not p["act"]: continue
            m = union(p["act"]); r = total(p["act"])
            key = f"{sub[:sub.rfind('.')+3]}.{pn}" if "." in sub else f"{sub}.{pn}"
            res["in_flight"].setdefault(key, []).append(r / total(m) if total(m) else 0.0)
        # 掩盖对
        for tag, (pa, pb, _) in CANONICAL.items():
            A, B = pipes.get(pa), pipes.get(pb)
            if not A or not B or not A["act"] or not B["act"]: continue
            ua, ub = union(A["act"]), union(B["act"])
            ta, tb = total(ua), total(ub)
            if ta <= 0 or tb <= 0: continue
            ov = inter(ua, ub) / min(ta, tb) * 100
            res["pairs"].setdefault(tag, []).append(ov)
    # 子核取中位数
    res["in_flight"] = {k: sorted(v)[len(v)//2] for k, v in res["in_flight"].items()}
    res["pairs"] = {k: round(sorted(v)[len(v)//2], 2) for k, v in res["pairs"].items()}
    # cv 双流水：CUBE（cubecore）与 VECTOR（各 veccore 并集）活跃区间跨子核重叠
    cube_act, vec_act = [], []
    for f in files:
        sub = os.path.basename(os.path.dirname(f))
        pipes = load_pipes(f)
        if sub.endswith("cubecore0") and pipes.get("CUBE") and pipes["CUBE"]["act"]:
            cube_act.extend(pipes["CUBE"]["act"])
        elif ".veccore" in sub and pipes.get("VECTOR") and pipes["VECTOR"]["act"]:
            vec_act.extend(pipes["VECTOR"]["act"])
    if cube_act and vec_act:
        ua, ub = union(cube_act), union(vec_act)
        ta, tb = total(ua), total(ub)
        if ta > 0 and tb > 0:
            res["pairs"]["CV_CUBE_VEC"] = round(inter(ua, ub) / min(ta, tb) * 100, 2)
    return res


def verdict(tr):
    """三分类双流水判定（≥30% 成功 / <30% 未成功；无数据=不适用）：
    - c 内：cube 子核 搬运×CUBE 计算 重叠 = max(C_MTE1_MAC, C_MTE2_MAC)
    - v 内：vector 子核 搬运×VECTOR 计算 重叠 = V_MTE2_VEC
    - cv  ：Cube 子核 × Vector 子核 并行重叠 = CV_CUBE_VEC（CVPipelining 目标）"""
    if not tr or not tr.get("pairs"):
        return {"level": "无数据", "dims": {}}
    pairs = tr["pairs"]
    dims = {}
    cube_pairs = [p for p in ("C_MTE1_MAC", "C_MTE2_MAC") if p in pairs]
    if cube_pairs:
        best = max(cube_pairs, key=lambda p: pairs[p])
        dims["c"] = {"overlap": pairs[best], "ok": pairs[best] >= 30}
    if "V_MTE2_VEC" in pairs:
        dims["v"] = {"overlap": pairs["V_MTE2_VEC"], "ok": pairs["V_MTE2_VEC"] >= 30}
    if "CV_CUBE_VEC" in pairs:
        dims["cv"] = {"overlap": pairs["CV_CUBE_VEC"], "ok": pairs["CV_CUBE_VEC"] >= 30}
    n_ok = sum(1 for d in dims.values() if d["ok"])
    lvl = "生效" if n_ok else "未生效"
    return {"level": lvl, "dims": dims}


def _lvl(ov):
    return "生效" if ov >= 30 else ("部分" if ov >= 10 else "未生效")


# BISHENG/HIVM CVPipelining & PlanMemory 落地标记（AscendNPU-IR/docs 权威口径）
BACKEND_MARKERS = {
    "cube_loop": "CVPipelining cube 工作项循环属性",
    "vector_loop": "CVPipelining vector 工作项循环属性",
    "multibuffer": "set-workspace-multibuffer 编译选项",
    "alloc_workspace": "PlanMemory CV 流 Workspace 分配",
    "local_alloc": "多缓冲 local 分配",
    "memdesc": "多缓冲 memdesc",
}
# 编译旋钮（cache <kernel>.json 元数据；比文本标记更可靠——多缓冲开关的后端真值）
COMPILE_KNOBS = ["multibuffer", "num_stages", "num_buffers_warp_spec",
                 "set_workspace_multibuffer", "enable_dynamic_cv_pipeline"]


def ir_side(ir_dir, kernel):
    """triton cache → 三层流水化证据。
    frontend_request: num_stages>1 循环数（前端请求）；
    backend_applied: CVPipelining/PlanMemory 落地标记命中清单；
    compile_knobs: <kernel>.json 编译旋钮（multibuffer 等后端开关真值）。"""
    out = {"frontend_num_stages_loops": 0, "max_stages": None,
           "backend_markers": {}, "compile_knobs": {}, "files": 0}
    if not ir_dir: return out
    files = glob.glob(os.path.join(ir_dir, "**", f"{kernel}.tta*"), recursive=True) + \
            glob.glob(os.path.join(ir_dir, "**", f"{kernel}.ttir"), recursive=True)
    out["files"] = len(files)
    for f in files:
        try: txt = open(f, encoding="utf-8", errors="replace").read()
        except Exception: continue
        for m in re.finditer(r"num_stages\s*=\s*(\d+)", txt):
            v = int(m.group(1))
            if v > 1:
                out["frontend_num_stages_loops"] += 1
                out["max_stages"] = max(out["max_stages"] or 0, v)
        for pat in BACKEND_MARKERS:
            if pat in txt:
                out["backend_markers"][pat] = out["backend_markers"].get(pat, 0) + txt.count(pat)
    # 编译旋钮（json 元数据，多变体取"最真值"：multibuffer 任一 True 即 True）
    for jf in glob.glob(os.path.join(ir_dir, "**", f"{kernel}.json"), recursive=True):
        try: meta = json.load(open(jf))
        except Exception: continue
        for k in COMPILE_KNOBS:
            v = meta.get(k)
            if v is None: continue
            if k == "multibuffer":
                out["compile_knobs"][k] = bool(out["compile_knobs"].get(k) or v)
            elif k not in out["compile_knobs"]:
                out["compile_knobs"][k] = v
    return out


def parse_last_pass(ir_dump_dir, kernel, suffix=""):
    """解析 skill IR 提取产物 <kernel>_last_pass<suffix>.mlir（bishengir 重放 + PlanMemory 后）：
    pointer_cast 字面字节偏移 = 缓冲地址层——同形 memref 的等距偏移群即多缓冲槽位。
    suffix="" 为 run_and_extract 默认口径（--enable-auto-multi-buffer=False）；
    suffix="_mb" 为 A/B 对照口径（True）——两者 diff 即双流水结构是否落地的 IR 证据。"""
    import re as _re
    f = os.path.join(ir_dump_dir, f"{kernel}_last_pass{suffix}.mlir")
    if not os.path.exists(f): return None
    # 全量 dump（双地址 pointer_cast 在中间 pass，last_pass 只有最终地址）
    ff = f + ".full"
    txt_full = open(ff, encoding="utf-8", errors="replace").read() if os.path.exists(ff) else None
    txt = open(f, encoding="utf-8", errors="replace").read()
    pc = _re.findall(r"pointer_cast\(%c(\d+)_i64\)\s*:\s*memref<([^,>]+),\s*#hivm\.address_space<(\w+)>>", txt)
    by_key = {}
    for off, shape, space in pc:
        by_key.setdefault((space, shape), set()).add(int(off))
    arenas = []
    for (space, shape), offs in sorted(by_key.items()):
        offs = sorted(offs)
        if len(offs) >= 2:
            deltas = sorted({b - a for a, b in zip(offs, offs[1:])})
            arenas.append({"space": space, "shape": shape, "offsets": offs,
                           "slots": len(offs), "stride_bytes": deltas[0] if len(deltas) == 1 else deltas[:3]})
    # 双地址 pointer_cast（多缓冲乒乓对）：同 op 双偏移 + loc 源码行——
    # "同一行代码的同一缓冲拿到两个地址"的直接表达（实测仅在 multibuffer=True 时出现）
    dual = _re.findall(r'pointer_cast\(%c(\d+)_i64,\s*%c(\d+)_i64\)\s*:\s*memref<([^,>]+),\s*#hivm\.address_space<(\w+)>>\s*loc\("([^"]+)":(\d+):', txt_full or txt)
    per_line = {}
    for a, b, shape, space, _f, ln in dual:
        k = (int(ln), shape, space)
        per_line.setdefault(k, []).append(int(b) - int(a))
    dual_lines = [{"line": k[0], "shape": k[1], "space": k[2], "pairs": len(v),
                   "delta_bytes": sorted(set(v))} for k, v in sorted(per_line.items())]
    return {"pointer_cast_count": len(pc), "arenas": arenas, "dual_addr_lines": dual_lines,
            "files": os.path.basename(f)}


def parse_pass_diff(ir_dump_dir, kernel):
    """解析 run_full_analysis 生成的 <kernel>_pass_diff.txt（diff_pass.sh 输出）：
    {pass名: '成功'/'未成功'}——pass 是否改写 IR 的最直接编译器证据。"""
    f = os.path.join(ir_dump_dir, f"{kernel}_pass_diff.txt")
    if not os.path.exists(f): return None
    res, cur = {}, None
    for line in open(f, encoding="utf-8", errors="replace"):
        if line.startswith("===== pass="):
            cur = line.strip()[len("===== pass="):]
        elif cur and line.startswith("变换:"):
            res[cur] = "成功" if "有" in line else "未成功"
    return res or None


def dual_flag_edges(simdir):
    """运行时乒乓签名（pc 级，严格口径）：
    **同一同步点（同一 pc 的 WAIT_FLAG）跨迭代交替使用 ≥2 个 FLAG id**
    = 一个代码位置对应多份缓冲（各自旗标）——双流水乒乓的直接证据。
    注意区分：仅"同一依赖边上有多个 flag"不成立（可能是多条数据流各自的
    单缓冲同步点）；必须同一 pc 交替才算。返回 {边: {pc: [flag ids]}}。"""
    import re
    pat = re.compile(r"PIPE:(\w+),TRIGGERPIPE:(\w+)(?:,FLAGID:(\d+))?")
    out = {}
    for f in sorted(glob.glob(os.path.join(simdir, "core0.*", "trace.json"))):
        pc_flags = {}
        pc_edge = {}
        for e in json.load(open(f))["traceEvents"]:
            if e.get("ph") != "X" or not str(e.get("name", "")).startswith("WAIT_FLAG"):
                continue
            m = pat.search(str((e.get("args") or {}).get("detail", "")))
            if not m: continue
            pc = (e.get("args") or {}).get("pc_addr")
            pc_flags.setdefault(pc, set()).add(m.group(3))
            pc_edge[pc] = f"{m.group(1)}<-{m.group(2)}"
        for pc, flags in pc_flags.items():
            if len(flags) >= 2:
                out.setdefault(pc_edge[pc], {})[pc] = sorted(flags)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="OPPROF_*/simulator 目录")
    ap.add_argument("--ir-dir", default=None, help="triton cache 目录（旋钮/标记检查）")
    ap.add_argument("--ir-dump-dir", default=None,
                    help="triton-latency-optimizer run_and_extract.sh 产物目录（<kernel>_last_pass.mlir 地址层）")
    ap.add_argument("--kernel", default=None, help="kernel 名（IR 文件匹配）")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr = trace_side(a.dir)
    vd = verdict(tr)
    ir = ir_side(a.ir_dir, a.kernel) if (a.ir_dir and a.kernel) else \
        {"frontend_num_stages_loops": None, "backend_markers": {}, "compile_knobs": {}}
    dfe = dual_flag_edges(a.dir) if a.dir else {}
    fe_on = bool(ir.get("frontend_num_stages_loops"))
    be_on = bool(ir.get("backend_markers"))
    mb_knob = bool((ir.get("compile_knobs") or {}).get("multibuffer"))
    n_dfe = sum(len(pcs) for pcs in dfe.values())
    dims = vd.get("dims") or {}
    parts = []
    for k, name in (("c", "c内"), ("v", "v内"), ("cv", "cv")):
        if k in dims:
            parts.append(f"{name}{'✅' if dims[k]['ok'] else '❌'}{dims[k]['overlap']}%")
    combined = "，".join(parts)
    fail_any = any(k in dims and not dims[k]["ok"] for k in ("c", "v", "cv"))
    if fail_any and mb_knob:
        combined += "（多缓冲已编译：旋钮 multibuffer=True；未成功项为重叠不足，非缓冲缺失）"
    out = {"trace": tr, "verdict": vd, "ir": ir, "dual_flag_edges": dfe, "combined": combined}
    # 地址层（last_pass.mlir，来自 skill IR 提取脚本；best-effort）
    lp = parse_last_pass(a.ir_dump_dir, a.kernel) if (a.ir_dump_dir and a.kernel) else None
    out["last_pass"] = lp
    # A/B 对照（multibuffer=True 重放）：结构 diff 判定双流水是否落地
    lp_mb = parse_last_pass(a.ir_dump_dir, a.kernel, "_mb") if (a.ir_dump_dir and a.kernel) else None
    out["last_pass_mb"] = lp_mb
    if lp and lp_mb:
        n_off = sum(x["slots"] for x in lp["arenas"])
        n_on = sum(x["slots"] for x in lp_mb["arenas"])
        out["mb_ab"] = {
            "slots_off": n_off, "slots_on": n_on,
            "ptr_cast_off": lp["pointer_cast_count"], "ptr_cast_on": lp_mb["pointer_cast_count"],
            "struct_changed": (n_off != n_on or lp["pointer_cast_count"] != lp_mb["pointer_cast_count"]),
        }
        ch = out["mb_ab"]["struct_changed"]
        print(f"地址层 A/B(multibuffer off→on): 槽位 {n_off}→{n_on}，"
              f"pointer_cast {lp['pointer_cast_count']}→{lp_mb['pointer_cast_count']} ⇒ "
              f"{'双流水结构落地（结构随旋钮变化）' if ch else '旋钮无结构效果（该 kernel 未受 multibuffer 影响）'}")
    # pass 级变换验证（diff_pass.sh 产物，最直接的编译器证据）
    out["pass_diff"] = parse_pass_diff(a.ir_dump_dir, a.kernel) if a.ir_dump_dir else None
    if out["pass_diff"]:
        print("pass 变换:", out["pass_diff"])

    if lp:
        arena_s = "; ".join(f"{x['space']}:{x['shape'].split('x')[0]}形×{x['slots']}槽(stride {x['stride_bytes']}B)"
                            for x in lp["arenas"][:3])
        print(f"地址层(last_pass): pointer_cast×{lp['pointer_cast_count']}"
              + (f"，槽位群: {arena_s}" if arena_s else ""))
    print(f"三维度: {combined}")
    if tr:
        print("掩盖对:", tr["pairs"])
        print("在途并发(raw/merged):", {k: round(v, 2) for k, v in tr["in_flight"].items()})
    print("IR 编译旋钮:", ir.get("compile_knobs"))
    print("双flag多缓冲边:", dfe if dfe else "无")
    print("综合判定:", combined)
    if a.json:
        json.dump(out, open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"[json] {a.json}")


if __name__ == "__main__":
    main()
