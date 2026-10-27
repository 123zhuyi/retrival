#!/usr/bin/env python3
"""阻塞点分析 v3（定义：某 pipe 执行时其他 pipe 都在等待 = 独占执行期）。
区间扫描实现：每 pipe 指针推进，O(S*P)。

v3 增强——源码行定位（skill 口径 6b/6c 的脚本化）：
  1. args.code 解析保留 file:line（v2 只取 basename 丢行号）；
     多行 code 为内联帧链，取首个非 internal 帧为主帧，其余计为内联帧数；
     internal:0 为编译器内部标签（标记展示，不作优化靶点）。
  2. 聚合输出 源文件:行 + 源码语句（--no-src 可关）。
  3. 新增「按等待行」与「执行→等待 联合定位」两张表（哪一行在等谁）。
  4. 非 kernel 主源文件的行标 ⚠外部（越界行隔离，口径 6c）。
"""
import json, argparse, os, re
from collections import defaultdict

WAIT_INSTR = {"WAIT_FLAG", "WAIT_FLAG_DEVI", "BAR"}
_INTERNAL = "internal:0"
_FILE_LINES = {}


def parse_code(code):
    """args.code → (primary_full_loc|None, inline_extra:int, internal_only:bool)"""
    lines = [l for l in (code or "").splitlines() if l]
    primary, extra = None, 0
    for l in lines:
        if l.endswith(_INTERNAL):
            continue
        if primary is None:
            primary = l
        else:
            extra += 1
    return primary, extra, (primary is None and bool(lines))


def loc_file_line(full):
    """abspath:line → (abspath, line:int) 或 (None, None)"""
    if not full:
        return None, None
    m = re.match(r"^(.*):(\d+)$", full)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def src_stmt(full, maxlen=56):
    """读取 abspath:line 的源码语句（缓存；不可读返回 ''）。"""
    path, line = loc_file_line(full)
    if not path:
        return ""
    try:
        if path not in _FILE_LINES:
            with open(path, encoding="utf-8", errors="replace") as f:
                _FILE_LINES[path] = f.readlines()
        ls = _FILE_LINES[path]
        if 1 <= line <= len(ls):
            s = ls[line - 1].strip()
            return s if len(s) <= maxlen else s[: maxlen - 1] + "…"
    except Exception:
        pass
    return ""


def fmt_loc(full, main_file, with_src=True):
    """格式化定位标签：basename:line[⚠外部]（+ 语句）。"""
    if not full:
        return "internal"
    path, line = loc_file_line(full)
    if not path:
        return os.path.basename(full)
    tag = f"{os.path.basename(path)}:{line}"
    if main_file and os.path.abspath(path) != main_file:
        tag += "⚠外部"
    if with_src:
        st = src_stmt(full)
        if st:
            tag += f"  | {st}"
    return tag


def merged(ivs):
    ivs = sorted((a, b, t) for a, b, t in ivs if b > a)
    out = []
    for a, b, t in ivs:
        if out and a <= out[-1][1]:
            o = out[-1]
            if b > o[1]:
                out[-1] = (o[0], b, t)
        else:
            out.append((a, b, t))
    return out


def subtract(waits, execs):
    """waits 区间去掉与 execs 的重叠，返回独占等待区间。"""
    out = []
    for a, b, t in waits:
        cur = a
        for ea, eb, _ in execs:
            if eb <= cur:
                continue
            if ea >= b:
                break
            if ea > cur:
                out.append((cur, min(ea, b)))
            cur = max(cur, eb)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def pipe_states(trace):
    """trace.json → {pipe: {"exec": raw [(a,b,loc,extra,internal)], "wait": raw [(a,b,(loc,extra))]}}"""
    d = json.load(open(trace))
    # pid → pipe 名（process_name 元数据；tid 是子队列号，不能作 pipe 分组键）
    pid2name = {}
    for e in d.get("traceEvents", []):
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid2name[str(e.get("pid"))] = e.get("args", {}).get("name", str(e.get("pid")))
    pipes = defaultdict(lambda: {"exec": [], "wait": []})
    for e in d.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        code = (e.get("args") or {}).get("code", "") or ""
        a, b = e["ts"], e["ts"] + e.get("dur", 0)
        pipe = pid2name.get(str(e.get("pid")), "?")
        if e.get("name") in WAIT_INSTR:
            loc, extra, _ = parse_code(code)
            pipes[pipe]["wait"].append((a, b, (loc, extra)))
        else:
            loc, extra, internal = parse_code(code)
            pipes[pipe]["exec"].append((a, b, loc, extra, internal))
    for p in pipes.values():
        p["exec"].sort(key=lambda x: (x[0], x[1]))
        p["wait"].sort(key=lambda x: (x[0], x[1]))
    return pipes


def main_source_file(trace):
    """kernel 主源文件 = exec/wait 事件占比最高的源码文件（越界行隔离基准，口径 6c）。"""
    d = json.load(open(trace))
    cnt = defaultdict(int)
    for e in d.get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        code = (e.get("args") or {}).get("code", "") or ""
        loc, _, _ = parse_code(code)
        path, _ = loc_file_line(loc)
        if path:
            cnt[os.path.abspath(path)] += 1
    return max(cnt.items(), key=lambda kv: kv[1])[0] if cnt else None


MAJOR_CUBECORE = {"CUBE", "MTE1", "MTE2", "FIXP"}   # cube 子核主要工作 pipe
MAJOR_VECCORE = {"VECTOR", "MTE2", "MTE3"}           # vector 子核主要工作 pipe
# FLOWCTRL/SCALAR/SCALARLDST/CACHEMISS 为调度/停泊/辅助 pipe，不参与阻塞判定


def analyze(trace, top=8, strict=True):
    """阻塞点聚合。

    strict=True（默认，严格口径）：某 pipe 执行期间，**本子核其余所有主要工作
    pipe 全部处于等待**才算阻塞点——即该 pipe 把整个子核的工作流水全部挡住；
    主要 pipe 若本切片空闲（无等待区间）则不满足"全部被堵"，该切片不计。
    strict=False 为旧宽松口径（≥1 pipe 在等）。
    按阻塞时长降序输出四张表。
    """
    pipes = pipe_states(trace)
    all_pipes = set(pipes)
    majors = (MAJOR_CUBECORE & all_pipes) or (MAJOR_VECCORE & all_pipes) or all_pipes
    P = {}
    for p, d in pipes.items():
        ex_raw = d["exec"]
        ex_m = merged([(a, b, None) for a, b, *_ in ex_raw])
        wa_m = merged([(a, b, None) for a, b, _ in d["wait"]])
        wa_only = subtract(wa_m, ex_m)
        P[p] = {"exec": ex_raw, "wait": wa_only, "wait_raw": d["wait"]}
    pts = set()
    for p in P.values():
        for a, b, *_ in p["exec"]:
            pts.add(a); pts.add(b)
        for a, b in p["wait"]:
            pts.add(a); pts.add(b)
    pts = sorted(pts)
    idx = {p: {"e": 0, "w": 0, "wr": 0} for p in P}
    # 第一遍：收集严格切片（按阻塞者分组，留时间戳供周期性判定）
    slices_by_blocker = defaultdict(list)   # (ep, eloc, extra) -> [(t0, dt, waits)]
    for i in range(len(pts) - 1):
        t0, t1 = pts[i], pts[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        execs, waits = [], []
        for p, st in P.items():
            ii = idx[p]
            while ii["e"] < len(st["exec"]) and st["exec"][ii["e"]][1] <= t0:
                ii["e"] += 1
            while ii["w"] < len(st["wait"]) and st["wait"][ii["w"]][1] <= t0:
                ii["w"] += 1
            if ii["e"] < len(st["exec"]) and st["exec"][ii["e"]][0] <= t0:
                _a, _b, loc, extra, _internal = st["exec"][ii["e"]]
                execs.append((p, loc, extra))
            elif ii["w"] < len(st["wait"]) and st["wait"][ii["w"]][0] <= t0:
                while ii["wr"] < len(st["wait_raw"]) and st["wait_raw"][ii["wr"]][1] <= t0:
                    ii["wr"] += 1
                wloc = None
                if ii["wr"] < len(st["wait_raw"]) and st["wait_raw"][ii["wr"]][0] <= t0:
                    wloc = st["wait_raw"][ii["wr"]][2][0]
                waits.append((p, wloc))
        if len(execs) == 1 and waits:
            ep, eloc, eextra = execs[0]
            if strict:
                # 严格：其余所有主要工作 pipe 必须全部在等（任一空闲即不满足"全部被堵"）
                waiting_pipes = {wp for wp, _ in waits}
                if not all(q in waiting_pipes for q in majors if q != ep):
                    continue
            slices_by_blocker[(ep, eloc, eextra)].append((t0, t1, waits))

    # 第二遍：同一阻塞者的相邻切片先合并成"阻塞事件"（同一步内被事件边界切碎的
    # 连续阻塞算一次），再按事件起点判周期——须重复 ≥3 次且间隔大致均匀
    # （一次性/启动期阻塞不算阻塞点；max_interval ≤ 2.5×median_interval）
    def is_periodic(ts):
        if len(ts) < 3:
            return False, 0.0
        ivs = [b - a for a, b in zip(ts, ts[1:])]
        med = sorted(ivs)[len(ivs) // 2]
        if med <= 0:
            return False, 0.0
        return max(ivs) <= med * 2.5 and min(ivs) >= med / 2.5, med

    agg_pipe = defaultdict(float)
    agg_loc = defaultdict(float)        # (pipe, loc, extra) → 周期阻塞总时长
    agg_wait = defaultdict(float)       # (pipe, wait_loc) → 同时长
    agg_joint = defaultdict(float)      # ((ep,eloc),(wp,wloc)) → 时长
    period_info = {}                    # (pipe, loc) → (事件数, 周期µs)
    for (ep, eloc, eextra), sls in slices_by_blocker.items():
        # 相邻切片合并为阻塞事件（间隔≤1e-3µs 视为同一事件）
        sls = sorted(sls, key=lambda x: x[0])
        eps = []  # (start, dur, waits_map wp→wloc)
        for t0, t1, waits in sls:
            if eps and t0 - eps[-1][0] - eps[-1][1] <= 1e-3:
                p0, pd0, pw = eps[-1]
                for wp, wloc in waits:
                    pw.setdefault(wp, wloc)
                eps[-1] = (p0, t1 - p0, pw)
            else:
                eps.append((t0, t1 - t0, {wp: wloc for wp, wloc in waits}))
        ts = [e[0] for e in eps]
        ok, med = is_periodic(ts)
        if not ok:
            continue
        n = len(eps)
        dur = sum(d for _, d, _ in eps)
        period_info[(ep, eloc)] = (n, med)
        agg_pipe[ep] += dur
        agg_loc[(ep, eloc, eextra)] += dur
        for t0, dt, waits in eps:
            for wp, wloc in waits.items():
                agg_wait[(wp, wloc)] += dt
                agg_joint[((ep, eloc), (wp, wloc))] += dt
    return {
        "sole_exec_by_pipe": sorted(agg_pipe.items(), key=lambda kv: -kv[1])[:top],
        "sole_exec_by_loc": sorted(agg_loc.items(), key=lambda kv: -kv[1])[:top],
        "wait_by_loc": sorted(agg_wait.items(), key=lambda kv: -kv[1])[:top],
        "joint": sorted(agg_joint.items(), key=lambda kv: -kv[1])[:top],
        "period_info": period_info,
    }


def pair_analysis(trace, threshold=10.0, top=6):
    """掩盖失效对（重叠度<阈值%）+ 每侧主导代码行（根因→代码位置）。"""
    pipes = pipe_states(trace)
    P = {p: [(a, b, loc) for a, b, loc, _, _ in d["exec"]] for p, d in pipes.items()}
    span = max((b for l in P.values() for a, b, _ in l), default=0.0)
    # 噪声管道过滤：活跃 <1% span 不进矩阵（与掩盖矩阵口径一致）
    names = sorted(p for p, l in P.items()
                   if sum(b - a for a, b, _ in l) > 0.01 * span)

    def top_locs(p, n=3):
        agg = defaultdict(float)
        for a, b, loc in P[p]:
            agg[loc] += b - a
        return sorted(agg.items(), key=lambda kv: -kv[1])[:n]

    import itertools
    out = []
    for a, b in itertools.combinations(names, 2):
        A = merged([(x, y, None) for x, y, _ in P[a]])
        B = merged([(x, y, None) for x, y, _ in P[b]])
        if not A or not B:
            continue
        ua = sum(x[1] - x[0] for x in A)
        ub = sum(x[1] - x[0] for x in B)
        if min(ua, ub) <= 0:
            continue
        i = j = 0
        inter = 0.0
        while i < len(A) and j < len(B):
            s = max(A[i][0], B[j][0])
            e = min(A[i][1], B[j][1])
            if e > s:
                inter += e - s
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        ov = 100.0 * inter / min(ua, ub)
        if ov < threshold:
            out.append((a, b, ov, top_locs(a), top_locs(b)))
    out.sort(key=lambda t: t[2])
    return out[:top]


def main_pairs(trace, threshold, with_src, main_file):
    print("== 掩盖失效对（活跃区间重叠度<阈值；每侧给主导代码行）==")
    for a, b, ov, la, lb in pair_analysis(trace, threshold):
        print(f"  {a} x {b}: 重叠 {ov:.1f}% ★")
        print(f"    {a:<10}: " + " / ".join(f"{fmt_loc(l, main_file, with_src)}({d:.0f}us)" for l, d in la))
        print(f"    {b:<10}: " + " / ".join(f"{fmt_loc(l, main_file, with_src)}({d:.0f}us)" for l, d in lb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--pairs", action="store_true", help="掩盖失效对 + 代码归因模式")
    ap.add_argument("--threshold", type=float, default=10.0)
    ap.add_argument("--no-src", action="store_true", help="不读取源码语句（大 trace 提速）")
    a = ap.parse_args()
    with_src = not a.no_src
    main_file = main_source_file(a.trace)
    if a.pairs:
        main_pairs(a.trace, a.threshold, with_src, main_file)
        return
    r = analyze(a.trace, a.top)
    mf = os.path.basename(main_file) if main_file else "?"
    print(f"== 阻塞点（严格：1 pipe 独占执行 + 其余主要 pipe 全等待 + 周期重复≥3次）==  [主源文件: {mf}]")
    print("-- 按执行 pipe：")
    for p, dur in r["sole_exec_by_pipe"]:
        print(f"   {p:<12} S{dur:9.1f}")
    print("-- 按执行 pipe + 代码行（源文件:行 + 语句）：")
    for (p, loc, extra), dur in r["sole_exec_by_loc"]:
        tag = fmt_loc(loc, main_file, with_src)
        if extra:
            tag += f" [+{extra}内联帧]"
        pinf = r.get("period_info", {}).get((p, loc))
        tail = f" S{dur:9.1f}" + (f"  ×{pinf[0]} 周期≈{pinf[1]:.2f}µs" if pinf else "")
        print(f"   {p:<12} {tag:<70}{tail}")
    print("-- 按等待 pipe + 等待行（哪一行在等）：")
    for (p, wloc), dur in r["wait_by_loc"]:
        print(f"   {p:<12} {fmt_loc(wloc, main_file, with_src):<70} S{dur:9.1f}")
    print("-- 执行→等待 联合定位（执行方行 ← 等待方行）：")
    for ((ep, eloc), (wp, wloc)), dur in r["joint"]:
        print(f"   {ep} @ {fmt_loc(eloc, main_file, with_src)}")
        print(f"      ← {wp} 等 @ {fmt_loc(wloc, main_file, with_src)}  S{dur:9.1f}")


if __name__ == "__main__":
    main()
