#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
"""pipe 间掩盖度分析（triton-profiling-analysis skill 自带分析脚本）。

消费 triton-simulator 仿真产物 trace.json（Chrome Trace Format），输出三张表：
  1. pipe 活跃表：每 pipe 事件数 / 原始忙碌和（>100% span = pipe 内在途并发，即双缓冲证据）
     / 归并后忙碌（墙钟活跃时间线）/ 占 span 比
  2. 掩盖矩阵：pipe 对重叠度 = 归并活跃区间交集 / 并集（时间轴 Jaccard），
     < 阈值（默认 10%，源自 profiling 方案气泡定位表）命中「掩盖失效」
  3. WAIT_FLAG 归因：等待方 ← 触发方（FLAGID）的次数与总等待时长

计算口径（与 references/simulator-bubble-analysis.md 一致）：
  - pipe 从 metadata（ph="M" process_name）动态发现，禁止硬编码
  - pipe 内并发区间先归并为互不相交集合再求交并——否则交集会按在途份数重复计数
  - 重叠度分母用并集：对称、有界 [0,1]，回答「任一 pipe 工作的总时间里两者同时工作的占比」

支持两种 trace 结构（自动识别）：
  - 子核文件（coreN.veccoreM/trace.json）：pid=数字 pipe ID（metadata 映射到名称），tid=线程
  - 合并文件（OPPROF_*/simulator/trace.json）：pid="coreN.子核N" 字符串，tid=pipe 名称
  掩盖度是子核内 pipe 对的属性——优先用子核文件；合并文件按 (pid,tid) 分组等价处理。

用法：
  python3 pipe_overlap_analysis.py <trace.json>                       # 单文件
  python3 pipe_overlap_analysis.py --dir <OPPROF_*/simulator>         # 批量：*/trace.json 逐子核 + 汇总
  选项：--threshold 10   掩盖失效判定阈值（%）
        --json <path>    机器可读结果落盘
        --top 8          WAIT_FLAG 归因显示条数

退出码：0 = 正常分析；2 = 输入不存在 / 无有效事件。
仅依赖 Python 标准库。
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import defaultdict

# PPT 气泡定位表中的 pipe 对 → 报告标签（存在才计算）
PPT_PAIRS = {
    frozenset(("MTE2", "MTE1")): "C_MTE2_MTE1 掩盖",
    frozenset(("MAC", "MTE1")): "C_MTE1_MAC 掩盖",
    frozenset(("MAC", "FIXPIPE")): "C_MAC_FIXPIPE 掩盖",
    frozenset(("VECTOR", "MTE2")): "V_MTE2_VEC 掩盖",
    frozenset(("VECTOR", "MTE3")): "V_VEC_MTE3 掩盖",
}
# V_C 掩盖（整体 C 流水 × V 流水）需跨子核聚合 C/V 侧全体 pipe，本脚本按子核内
# pipe 对输出，V_C 由分析者基于各子核结果聚合，不在此自动判定。

WAIT_DETAIL = re.compile(r"PIPE:(?P<pipe>[A-Za-z0-9]+),TRIGGERPIPE:(?P<trig>[A-Za-z0-9]+)(?:,FLAGID:(?P<flag>\d+))?")


def merge_intervals(iv):
    """区间归并为互不相交集合（pipe 内并发折叠成墙钟时间线）。"""
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return out


def total_len(iv):
    return sum(b - a for a, b in iv)


def intersect_len(A, B):
    """双指针求两组互不相交区间的交集总时长。"""
    i = j = 0
    s = 0.0
    while i < len(A) and j < len(B):
        a = A[i][0] if A[i][0] > B[j][0] else B[j][0]
        b = A[i][1] if A[i][1] < B[j][1] else B[j][1]
        if b > a:
            s += b - a
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return s


def load_events(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("displayTimeUnit", "?"), d.get("traceEvents", [])


def discover_pipes(events):
    """metadata（ph=M process_name）→ pid 映射。返回 pid→pipe 名（无映射的 pid 原样）。"""
    m = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            args = e.get("args") or {}
            if "pid" in e and isinstance(args, dict) and args.get("name"):
                m[e["pid"]] = args["name"]
    return m


WAIT_FLAG_PREFIX = "WAIT_FLAG"  # 空等指令前缀：覆盖 WAIT_FLAG / WAIT_FLAG_DEVI / WAIT_FLAG_DEV
                                # 等设备级变体（驻留 FLOWCTRL 队列）；ph=X 无法区分，须按 name 剔
SYNC_BLOCK_INSTR = {"BAR"}  # barrier 汇合（精确匹配，非计算）


def is_wait_instr(name):
    """空等指令判定：WAIT_FLAG 前缀匹配（含 _DEVI/_DEV 设备级变体）。

    实测 200_GDN_vllm trace：WAIT_FLAG_DEVI 占总等待约一半（state 50.2µs vs
    WAIT_FLAG 51.1µs），精确匹配会把它漏进 eff 口径（FLOWCTRL eff 虚高至 95%）。
    """
    return isinstance(name, str) and name.startswith(WAIT_FLAG_PREFIX)


def group_events(events):
    """识别 trace 结构并按 (子核, pipe) 分组 X 事件。

    返回 (groups, merged_schema, wait_groups, bar_groups, eff_groups)：
      groups:      {(subcore, pipe): [[ts, end], ...}  全部 X 事件（物理队列口径）
      wait_groups: 同结构，仅 name==WAIT_FLAG 的空等事件
      bar_groups:  同结构，仅 name==BAR 的 barrier 汇合事件
      eff_groups:  同结构，剔除 WAIT_FLAG 与 BAR 后的事件（有效计算口径）
      子核文件：subcore="self"，pipe=metadata 映射后的 pid
      合并文件：subcore=pid（coreN.子核N），pipe=tid
    """
    pid2name = discover_pipes(events)
    groups = defaultdict(list)
    wait_groups = defaultdict(list)
    bar_groups = defaultdict(list)
    eff_groups = defaultdict(list)
    merged_schema = False
    for e in events:
        if e.get("ph") != "X" or not isinstance(e.get("ts"), (int, float)):
            continue
        pid = e.get("pid")
        if isinstance(pid, str) and "." in pid:  # 合并文件：pid=coreN.veccoreN
            merged_schema = True
            subcore, pipe = pid, str(e.get("tid"))
        else:
            subcore, pipe = "self", pid2name.get(pid, str(pid))
        ts = float(e["ts"])
        dur = float(e.get("dur") or 0)
        iv = [ts, ts + dur]
        key = (subcore, pipe)
        groups[key].append(iv)
        name = e.get("name")
        if is_wait_instr(name):
            wait_groups[key].append(list(iv))
        elif name in SYNC_BLOCK_INSTR:  # BAR
            bar_groups[key].append(list(iv))
        else:
            eff_groups[key].append(iv)
    return groups, merged_schema, wait_groups, bar_groups, eff_groups


def analyze_group(intervals, events_of_pipe, span, wait_intervals=None, eff_intervals=None,
                  bar_intervals=None):
    """单个 (子核, pipe) 的统计（四个口径）。

    merged_pct：物理队列占用（含 WAIT_FLAG/BAR）→ 空泡率 = 100 − merged_pct
    wait_pct  ：其中 WAIT_FLAG 空等占比（逻辑空泡）
    bar_pct   ：其中 BAR barrier 汇合占比（同步，非计算）
    eff_pct   ：剔除 WAIT_FLAG 与 BAR 的有效活跃（有效计算口径）
    """
    merged = merge_intervals(intervals)
    wait_m = merge_intervals(wait_intervals or [])
    eff_m = merge_intervals(eff_intervals if eff_intervals is not None else intervals)
    bar_m = merge_intervals(bar_intervals or [])
    return {
        "events": events_of_pipe,
        "raw_busy": total_len(intervals),
        "merged_busy": total_len(merged),
        "merged_pct": (total_len(merged) / span * 100) if span > 0 else 0.0,
        "wait_busy": total_len(wait_m),
        "wait_pct": (total_len(wait_m) / span * 100) if span > 0 else 0.0,
        "bar_busy": total_len(bar_m),
        "bar_pct": (total_len(bar_m) / span * 100) if span > 0 else 0.0,
        "eff_busy": total_len(eff_m),
        "eff_pct": (total_len(eff_m) / span * 100) if span > 0 else 0.0,
        "merged": merged,
        "wait_merged": wait_m,
        "bar_merged": bar_m,
        "eff_merged": eff_m,
    }


def wait_flag_attribution(events, span):
    """WAIT_FLAG 事件按 detail（等待方←触发方, FLAGID）归因。

    注意：事件自身 pid 与等待方/触发方不保证一致，一律以 detail 字符串为准。
    返回 [(等待方, 触发方, flag, 次数, 总时长, 占span%)] 按 总时长 降序。
    """
    agg = defaultdict(lambda: [0, 0.0])
    uimm_re = re.compile(r"UIMM:(0x[0-9a-fA-F]+)")
    for e in events:
        if e.get("ph") != "X" or not is_wait_instr(e.get("name")):
            continue
        detail = str((e.get("args") or {}).get("detail", ""))
        m = WAIT_DETAIL.search(detail)
        if m:
            key = (m.group("pipe"), m.group("trig"), m.group("flag"))
        else:
            # 设备级变体（WAIT_FLAG_DEVI/_DEV）：detail 只有 UIMM flag 掩码，
            # 无 pipe 归因信息——以指令名+UIMM 分组单列，不归入 "?"
            um = uimm_re.search(detail)
            key = (e.get("name"), "(device)", um.group(1) if um else "?")
        agg[key][0] += 1
        agg[key][1] += float(e.get("dur") or 0)
    rows = [(k[0], k[1], k[2], v[0], v[1], (v[1] / span * 100) if span > 0 else 0.0)
            for k, v in agg.items()]
    return sorted(rows, key=lambda r: -r[4])


def select_first_core_subcores(subcore_names):
    """自适应选取第一个核心（core0）的子核：1 个 cubecore + 至多 2 个 veccore。

    不同算子类型采集到的子核/pipe 构成不同（vector 型无 cubecore、cube 型无
    veccore、mix 型 1C+2V 全有，pipe 个数随之变化）——按实际存在的 core0.*
    子核选取，不假设固定数量。返回排序后列表（cube 在前，vec 按序号）。
    """
    def key(name):
        kind = 0 if "cubecore" in name else (1 if "veccore" in name else 2)
        return (kind, name)
    return sorted([s for s in subcore_names if s.startswith("core0.")], key=key)


def op_type_hint(subcore_names):
    """由子核构成推断算子类型（仅提示用）。"""
    has_c = any("cubecore" in s for s in subcore_names)
    has_v = any("veccore" in s for s in subcore_names)
    if has_c and has_v:
        return "mix 型（1C+2V）"
    return "cube 型" if has_c else "vector 型"


PIPE_ALIAS = {"VEC": "VECTOR"}  # detail 缩写 → flow cat 全名


def dependency_graph(events):
    """从 trace.json 提取 pipe 间依赖边（三源证据交叉验证）。

    证据源（同一条依赖边 A 依赖 B 的三个视角）：
      1. flow 事件（ph="s"/"t"，按 id 配对）：cat="AToB" 语义为 **A 依赖 B**
         （A 等 B 置 flag；s 事件在等待方 A，t 事件在触发方 B），两端 ts 差即
         依赖传播时延；
      2. WAIT_FLAG 指令（等待方视角）：args.detail "PIPE:A,TRIGGERPIPE:B,FLAGID:n"；
      3. SET_FLAG 指令（置位方视角）：detail 同编码。
    返回行列表：[(等待方, 触发方, flow数, 时延p50, 时延max, WAIT数, WAIT时长, SET数)]
    按 WAIT 时长降序（无等待时按 flow 数）。
    """
    src, dst = {}, {}
    wf = defaultdict(lambda: [0, 0.0])
    sf = defaultdict(int)
    for e in events:
        ph = e.get("ph")
        if ph == "s":
            src[e["id"]] = (e.get("cat"), e["ts"])
        elif ph == "t":
            dst[e["id"]] = (e.get("cat"), e["ts"])
        elif ph == "X" and (is_wait_instr(e.get("name")) or e.get("name") == "SET_FLAG"):
            m = WAIT_DETAIL.search(str((e.get("args") or {}).get("detail", "")))
            if not m:
                continue
            edge = (PIPE_ALIAS.get(m.group("pipe"), m.group("pipe")),
                    PIPE_ALIAS.get(m.group("trig"), m.group("trig")))
            if is_wait_instr(e["name"]):
                wf[edge][0] += 1
                wf[edge][1] += float(e.get("dur") or 0)
            else:
                sf[edge] += 1
    lat = defaultdict(list)
    for i, (cat, ts) in src.items():
        if i in dst and cat and dst[i][0] == cat:
            a, _, b = cat.partition("To")
            if b:
                lat[(PIPE_ALIAS.get(a, a), PIPE_ALIAS.get(b, b))].append(dst[i][1] - ts)
    edges = set(lat) | set(wf) | set(sf)
    rows = []
    for a, b in edges:
        v = sorted(lat.get((a, b), []))
        rows.append((a, b, len(v),
                     v[len(v) // 2] if v else None,
                     v[-1] if v else None,
                     wf.get((a, b), (0, 0.0))[0],
                     wf.get((a, b), (0, 0.0))[1],
                     sf.get((a, b), 0)))
    return sorted(rows, key=lambda r: (-r[6], -r[2]))


def fmt_deps(rows, span):
    print(f"{'依赖边(等待方 依赖 触发方)':<30}{'flow数':>7}{'时延p50':>9}{'时延max':>9}"
          f"{'WAIT数':>7}{'等待时长':>10}{'占span%':>8}{'SET数':>6}")
    for a, b, n, p50, mx, wcnt, wdur, scnt in rows:
        p50_s = f"{p50:.3f}" if p50 is not None else "-"
        mx_s = f"{mx:.3f}" if mx is not None else "-"
        pct = f"{wdur / span * 100:7.1f}" if span > 0 else "-"
        print(f"{a + ' 依赖 ' + b:<30}{n:>7}{p50_s:>9}{mx_s:>9}{wcnt:>7}{wdur:>10.2f}{pct:>8}{scnt:>6}")


def complement_intervals(busy, a, b):
    """归并区间的补集（[a,b] 内的空泡段）。"""
    gaps = []
    cur = a
    for s, e in busy:
        if s > cur:
            gaps.append([cur, min(s, b)])
        cur = max(cur, e)
        if cur >= b:
            break
    if cur < b:
        gaps.append([cur, b])
    return [g for g in gaps if g[1] - g[0] > 1e-12]


def analyze_trace(path, threshold, top, min_activity=1.0):
    _, events = load_events(path)
    groups, merged_schema, wait_groups, bar_groups, eff_groups = group_events(events)
    if not groups:
        return None
    X = [e for e in events if e.get("ph") == "X" and isinstance(e.get("ts"), (int, float))]
    tmin = min(e["ts"] for e in X)
    span = max(e["ts"] + float(e.get("dur") or 0) for e in X) - tmin

    result = {"file": path, "span": span, "t0": tmin, "t1": tmin + span,
              "merged_schema": merged_schema,
              "subcores": {}, "waits": wait_flag_attribution(events, span)}

    for (subcore, pipe), iv in groups.items():
        key = (subcore, pipe)
        result["subcores"].setdefault(subcore, {})[pipe] = analyze_group(
            iv, len(iv), span, wait_groups.get(key), eff_groups.get(key, iv),
            bar_groups.get(key))

    # 自适应聚焦第一个核心（1C+2V）：合并文件含全部子核，只保留 core0.*——
    # 子核/pipe 构成随算子类型变化（vector 无 cube、cube 无 vec、mix 全有），
    # 按实际存在的 core0 子核选取；子核文件本身就是单子核，无需过滤。
    if merged_schema:
        keep = select_first_core_subcores(result["subcores"].keys())
        dropped = len(result["subcores"]) - len(keep)
        result["subcores"] = {s: result["subcores"][s] for s in keep}
        result["dropped_subcores"] = dropped

    # 掩盖矩阵（v3 口径）：活跃区间 = X 事件剔除等待类（WAIT_FLAG/BAR）后的归并区间；
    # 重叠度/覆盖率一律基于活跃区间（eff_merged）——等待指令占队列但非工作，
    # 用队列占用口径会把「双方同时在等」误计为掩盖。
    # 过滤近零活跃 pipe：CACHEMISS/FLOWCTRL 等辅助队列时长≈0，防分母失真误报。
    pair_rows = []
    for subcore, pipes in result["subcores"].items():
        active = [(p, s["eff_merged"]) for p, s in pipes.items()
                  if span > 0 and s["eff_busy"] / span * 100 >= min_activity]
        for (pa, A), (pb, B) in itertools.combinations(sorted(active), 2):
            inter = intersect_len(A, B)
            la, lb = total_len(A), total_len(B)
            # v4 口径：分别以两 pipe 活跃为分母取较大值 = 交集/min(活跃A,活跃B)
            # ——「用时较少一方的活跃被对方覆盖的比例」，避免并集分母惩罚短 pipe
            ov = (inter / min(la, lb) * 100) if min(la, lb) > 0 else 0.0
            pair_rows.append({
                "subcore": subcore, "pipe_a": pa, "pipe_b": pb,
                "overlap_pct": ov,
                "label": PPT_PAIRS.get(frozenset((pa, pb)), ""),
                "hit": ov < threshold,
            })
    result["pairs"] = pair_rows
    return result


def gap_attribution(pipes, span, t0, t1, min_activity=1.0, top_n=3):
    """空泡归因（v3 口径，全部基于活跃区间）。

    空泡(P) = 活跃区间(P) 在 [t0,t1] 内的补集（含物理空闲 + 等待指令驻留）；
    覆盖率(Q, 空泡) = |活跃(Q) ∩ 空泡| / |空泡| —— 只统计他人真干活的时间。
    返回行：[(pipe, 空泡%, [(他人, 覆盖%)] top_n, 全局真空闲%)]，按空泡% 降序。
    """
    act = {p: s["eff_merged"] for p, s in pipes.items()
           if span > 0 and s["eff_busy"] / span * 100 >= min_activity}
    rows = []
    for p, s in pipes.items():
        if p not in act:
            continue
        gaps = complement_intervals(s["eff_merged"], t0, t1)
        bubble = total_len(gaps)
        if bubble <= 0:
            continue
        others = {q: intersect_len(gaps, a) / bubble * 100
                  for q, a in act.items() if q != p}
        union_others = merge_intervals([iv for q, a in act.items() if q != p for iv in a])
        global_idle = (bubble - intersect_len(gaps, union_others)) / bubble * 100
        top = sorted(others.items(), key=lambda kv: -kv[1])[:top_n]
        rows.append((p, bubble / span * 100 if span > 0 else 0.0,
                     [(q, c) for q, c in top if c > 1.0], global_idle))
    return sorted(rows, key=lambda r: -r[1])


def bubble_cause_rows(waits, pipes, span, t0, t1, min_activity=1.0):
    """每 pipe 空泡率与成因（默认流程第 3 段）。

    综合：① 空泡 = 100−活跃%（分解为物理空闲 + 队列等待驻留）
          ② 逻辑等待边（以 detail 的 PIPE: 为等待方——注意 WAIT_FLAG 事件记录在
             触发方队列，「P 等谁」与「谁等 P」都要看）
          ③ 空泡期间他人活跃覆盖率（gap_attribution）
    返回 [(pipe, 空泡%, 物理%, 驻留%, P等谁(top), 谁等P(top), 覆盖(top), 成因)]。
    """
    from collections import defaultdict as _dd
    by_waiter = _dd(lambda: _dd(float))   # P(等待方) -> 触发方 -> dur
    by_trigger = _dd(lambda: _dd(float))  # P(触发方) -> 等待方 -> dur
    for wpipe, trig, _flag, _cnt, dur, _pct in (waits or []):
        w = PIPE_ALIAS.get(wpipe, wpipe)   # detail 缩写归一（VEC→VECTOR）
        t = PIPE_ALIAS.get(trig, trig)
        by_waiter[w][t] += dur
        by_trigger[t][w] += dur
    rows = []
    for p, bpc, cov, gid in gap_attribution(pipes, span, t0, t1, min_activity):
        s = pipes.get(p)
        if not s:
            continue
        phys = max(0.0, 100.0 - s["merged_pct"])
        wb = by_waiter.get(p)
        wt = by_trigger.get(p)
        wait_by = f"{max(wb, key=wb.get)} {max(wb.values()):.1f}" if wb else "—"
        waited = f"{max(wt, key=wt.get)} {max(wt.values()):.1f}" if wt else "—"
        cov_s = "；".join(f"{q} {c:.0f}%" for q, c in cov) or "无"
        # 成因合成规则
        if bpc < 5:
            cause = "近饱和，无显著空泡"
        elif wb and max(wb.values()) / span * 100 > 15:
            t, dv = max(wb.items(), key=lambda kv: kv[1])
            cause = f"等待 {t} 主导（依赖链）"
        elif wt and max(wt.values()) / span * 100 > 15:
            w, dv = max(wt.items(), key=lambda kv: kv[1])
            cause = f"被依赖方：{w} 在等 P（P 是瓶颈供给）"
        elif phys > s["wait_pct"]:
            top_q = cov[0][0] if cov else "—"
            cause = f"物理空闲为主（期间 {top_q} 活跃：节拍/碎片）"
        else:
            cause = "等待驻留混合"
        rows.append((p, bpc, phys, s["wait_pct"], wait_by, waited, cov_s, cause))
    return sorted(rows, key=lambda r: -r[1])


def fmt_subcore(name, pipes, span, threshold, top_waits, min_activity=1.0, t0=None, t1=None,
                waits=None):
    print(f"\n=== 子核 {name} | span={span:.2f} ===")
    print(f"{'pipe':<12}{'events':>8}{'raw_busy%':>11}{'merged%':>9}{'wait%':>8}{'bar%':>7}{'活跃%':>8}")
    print(f"{'':12}{'':8}{'(含并发)':>11}{'(队列占用)':>9}{'(空等)':>8}{'(BAR)':>7}{'(剔等待)':>8}")
    for p, s in sorted(pipes.items(), key=lambda kv: -kv[1]["merged_busy"]):
        print(f"{p:<12}{s['events']:>8}{s['raw_busy']/span*100:>10.1f}%{s['merged_pct']:>8.1f}%"
              f"{s['wait_pct']:>7.1f}%{s['bar_pct']:>6.1f}%{s['eff_pct']:>7.1f}%")
    active = [(p, s) for p, s in pipes.items()
              if span > 0 and s["eff_busy"] / span * 100 >= min_activity]
    print(f"-- 掩盖矩阵（活跃>={min_activity:.0f}%；活跃区间=剔等待类；重叠度=交集/min(活跃)，"
          f"<{threshold}% 命中失效）--")
    for (pa, sa), (pb, sb) in itertools.combinations(sorted(active), 2):
        inter = intersect_len(sa["eff_merged"], sb["eff_merged"])
        m = min(sa["eff_busy"], sb["eff_busy"])  # v4：max(inter/A, inter/B) 的分母
        ov = (inter / m * 100) if m > 0 else 0.0
        tag = PPT_PAIRS.get(frozenset((pa, pb)), "")
        verdict = "★掩盖失效" if ov < threshold else "掩盖成立"
        print(f"  {pa:<10} x {pb:<10} {ov:>6.2f}%  {verdict}  {('['+tag+']') if tag else ''}")
    if t0 is not None:
        print(f"-- 每 pipe 空泡率与成因（空泡=活跃补集=物理空闲+等待驻留；覆盖率=他人活跃∩空泡）--")
        print(f"   {'pipe':<11}{'空泡%':>6}{'物理%':>7}{'驻留%':>7}  {'P等谁(µs)':<16}{'谁等P(µs)':<16}{'空泡期间活跃':<20}成因")
        for p, bpc, phys, res, wby, wted, cov_s, cause in bubble_cause_rows(
                waits, pipes, span, t0, t1, min_activity):
            print(f"   {p:<11}{bpc:>5.1f}%{phys:>6.1f}%{res:>6.1f}%  {wby:<16}{wted:<16}{cov_s:<20}{cause}")


def export_intervals(path, out_path, max_intervals=0):
    """将各 pipe 的原始/归并活跃区间导出为 JSON，供人工对照原始 trace.json 验证。

    结构：{file, merged_schema, pipes: {<pipe>: {pid, num_raw, raw_intervals,
    num_merged, merged_intervals, merged_busy}}}。raw_intervals 按 ts 升序，
    每条 [ts, ts+dur] 与一条 ph=X 事件一一对应。
    max_intervals > 0 时每 pipe 只保留前 N 条（统计字段仍是全量），文件更小便于人工翻阅。
    """
    _, events = load_events(path)
    pid2name = discover_pipes(events)
    name2pid = {v: k for k, v in pid2name.items()}
    groups, merged_schema, wait_groups, bar_groups, eff_groups = group_events(events)
    data = {"file": path, "merged_schema": merged_schema,
            "note": "raw_intervals 与 ph=X 事件一一对应 [ts, ts+dur]；"
                    "merged_intervals 为归并后互不相交活跃时间线；"
                    "wait_intervals/effective_intervals 为按 name==WAIT_FLAG 拆分的"
                    "空等/有效计算子集（归并后）"
                    + (f"；每 pipe 仅保留前 {max_intervals} 条（num_* 为全量）" if max_intervals else ""),
            "pipes": {}}
    for (subcore, pipe), iv in sorted(groups.items()):
        key = pipe if not merged_schema else f"{subcore}:{pipe}"
        gkey = (subcore, pipe)
        m = merge_intervals(iv)
        wm = merge_intervals(wait_groups.get(gkey, []))
        em = merge_intervals(eff_groups.get(gkey, iv))
        bm = merge_intervals(bar_groups.get(gkey, []))
        raw_sorted = sorted([a, b] for a, b in iv)
        data["pipes"][key] = {
            "pid": name2pid.get(pipe) if not merged_schema else subcore,
            "num_raw": len(iv),
            "raw_intervals": raw_sorted[:max_intervals] if max_intervals else raw_sorted,
            "num_merged": len(m),
            "merged_intervals": m[:max_intervals] if max_intervals else m,
            "merged_busy": round(total_len(m), 6),
            "bar_busy": round(total_len(bm), 6),
            "num_wait": len(wait_groups.get(gkey, [])),
            "wait_intervals": (wm[:max_intervals] if max_intervals else wm),
            "wait_busy": round(total_len(wm), 6),
            "num_effective": len(eff_groups.get(gkey, [])),
            "effective_intervals": (em[:max_intervals] if max_intervals else em),
            "effective_busy": round(total_len(em), 6),
        }
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    total_raw = sum(p["num_raw"] for p in data["pipes"].values())
    print(f"[导出] {out_path}")
    print(f"  pipes={len(data['pipes'])} 原始区间共 {total_raw} 条"
          + (f"（每 pipe 截前 {max_intervals} 条）" if max_intervals else "") + "；逐 pipe：")
    for k, p in sorted(data["pipes"].items(), key=lambda kv: -kv[1]["num_raw"]):
        print(f"    {k:<12} raw={p['num_raw']:>6} merged={p['num_merged']:>6} "
              f"busy={p['merged_busy']:>10} (wait={p['wait_busy']:>9} eff={p['effective_busy']:>10})")


def show_intervals(path, selected, max_show):
    """打印各 pipe 的活跃区间中间数据，供人工对照原始 trace.json 验证。

    每 pipe 输出：pid 映射、源事件样例（前3条原文）、原始区间（按 ts 排序）、
    归并区间及归并总长。合并文件结构取首个子核展示。
    """
    _, events = load_events(path)
    pid2name = discover_pipes(events)
    name2pid = {v: k for k, v in pid2name.items()}
    groups, merged_schema, _wg, _bg, _eg = group_events(events)
    if merged_schema:
        subcores = sorted({k[0] for k in groups})
        target = subcores[0]
        pipe_groups = {k[1]: v for k, v in groups.items() if k[0] == target}
        print(f"[提示] 合并文件结构：展示首个子核 {target}")
    else:
        target = "self"
        pipe_groups = {k[1]: v for k, v in groups.items() if k[0] == "self"}
    selected_set = set(selected) if selected else None
    pipes = [p for p in sorted(pipe_groups) if selected_set is None or p in selected_set]
    if not pipes:
        avail = ", ".join(sorted(pipe_groups))
        print(f"[警告] 未匹配到 pipe（可选：{avail}）", file=sys.stderr)
        return
    # 每个入选 pipe 收集前 3 条源 X 事件原文
    sample = defaultdict(list)
    for e in events:
        if e.get("ph") != "X" or not isinstance(e.get("ts"), (int, float)):
            continue
        if merged_schema and isinstance(e.get("pid"), str) and "." in e["pid"]:
            sc, pipe = e["pid"], str(e.get("tid"))
        else:
            sc, pipe = "self", pid2name.get(e.get("pid"), str(e.get("pid")))
        if sc == target and pipe in pipes and len(sample[pipe]) < 3:
            sample[pipe].append(e)
    for pipe in pipes:
        iv = pipe_groups.get(pipe) or []
        m = merge_intervals(iv)
        pid_disp = name2pid.get(pipe, "tid") if not merged_schema else "tid"
        print(f"\n== pipe {pipe}（pid={pid_disp} @ {target}）==")
        print(f"原始区间 {len(iv)} 条（每条 X 事件 [ts, ts+dur]）→ 归并后 {len(m)} 条，"
              f"归并总长 {total_len(m):.4f}")
        print("  源事件样例（前 3 条，字段与 trace.json 原文一致，可直接对照）:")
        for e in sample.get(pipe, []):
            print(f"    {{\"ph\": \"X\", \"pid\": {e.get('pid')!r}, \"tid\": {e.get('tid')!r}, "
                  f"\"ts\": {e.get('ts')}, \"dur\": {e.get('dur')}, \"name\": {e.get('name')!r}}}")
        srt = sorted(iv)
        head = ", ".join(f"[{a:.4f},{b:.4f})" for a, b in srt[:max_show])
        print(f"  原始区间（ts 升序）前 {min(max_show, len(srt))} 条: {head}")
        head_m = ", ".join(f"[{a:.4f},{b:.4f})" for a, b in m[:max_show])
        print(f"  归并区间前 {min(max_show, len(m))} 条: {head_m}")


def main():
    ap = argparse.ArgumentParser(description="pipe 间掩盖度分析（triton-profiling-analysis）")
    ap.add_argument("trace", nargs="?", help="单个 trace.json（子核文件或合并文件）")
    ap.add_argument("--dir", help="OPPROF_*/simulator 目录：逐子核分析 + 跨子核汇总")
    ap.add_argument("--threshold", type=float, default=10.0, help="掩盖失效阈值%%（默认 10）")
    ap.add_argument("--min-activity", type=float, default=1.0,
                    help="进入掩盖矩阵的最小 pipe 活跃度%%（默认 1，过滤 CACHEMISS 等噪声队列）")
    ap.add_argument("--json", help="机器可读结果落盘路径")
    ap.add_argument("--top", type=int, default=8, help="WAIT_FLAG 归因显示条数")
    ap.add_argument("--show-intervals", nargs="*", metavar="PIPE",
                    help="打印所选 pipe 的原始/归并活跃区间与源事件样例供人工核对"
                         "（不选 pipe 则全部；取输入清单首个文件）")
    ap.add_argument("--export-intervals", metavar="PATH",
                    help="将各 pipe 完整原始/归并区间导出为 JSON（人工核对用；取输入清单首个文件）")
    ap.add_argument("--max-intervals", type=int, default=0,
                    help="--export-intervals 每 pipe 保留条数（默认 0=全量；>0 时仅保留前 N 条，文件更小）")
    ap.add_argument("--max-show", type=int, default=10, help="--show-intervals 每段显示条数（默认 10）")
    ap.add_argument("--deps", action="store_true",
                    help="输出 pipe 间依赖图：flow(s/t 按 id 配对,cat=等待方To触发方) × "
                         "WAIT_FLAG(等待方) × SET_FLAG(置位方) 三源交叉 + 传播时延"
                         "（取输入清单首个文件）")
    args = ap.parse_args()
    if not args.trace and not args.dir:
        ap.error("需要 <trace.json> 或 --dir <dir>")

    files = []
    if args.dir:
        found = sorted(glob.glob(os.path.join(args.dir, "*", "trace.json")))
        if not found:
            print(f"[错误] {args.dir} 下未发现 */trace.json（子核文件）", file=sys.stderr)
            sys.exit(2)
        # 自适应聚焦第一个核心（1C+2V）：只取 core0.* 子核——
        # 子核/pipe 构成随算子类型变化（vector 型无 cubecore、cube 型无 veccore、
        # mix 型全有），按实际存在者选取，不假设固定个数。
        sub_names = [os.path.basename(os.path.dirname(f)) for f in found]
        keep = select_first_core_subcores(sub_names)
        if not keep:
            print(f"[错误] {args.dir} 下未发现 core0.* 子核（现有示例：{sub_names[:3]}…）",
                  file=sys.stderr)
            sys.exit(2)
        files = [f for f, s in zip(found, sub_names) if s in keep]
        print(f"[自适应] 选中 core0 子核 {len(files)}/{len(found)}："
              f"{'、'.join(keep)}（{op_type_hint(keep)}）")
    else:
        if not os.path.isfile(args.trace):
            print(f"[错误] 文件不存在: {args.trace}", file=sys.stderr)
            sys.exit(2)
        files = [args.trace]

    if args.show_intervals is not None:
        show_intervals(files[0], args.show_intervals, args.max_show)
        return
    if args.deps:
        _, events = load_events(files[0])
        X = [e for e in events if e.get("ph") == "X" and isinstance(e.get("ts"), (int, float))]
        span = (max(e["ts"] + float(e.get("dur") or 0) for e in X) - min(e["ts"] for e in X)) if X else 0
        print(f"pipe 依赖图 @ {files[0]}（span={span:.2f}）：")
        fmt_deps(dependency_graph(events), span)
        return
    if args.export_intervals:
        export_intervals(files[0], args.export_intervals, args.max_intervals)
        return

    all_results = []
    pair_collect = defaultdict(list)
    wait_collect = defaultdict(lambda: [0, 0.0])
    pipe_collect = defaultdict(lambda: defaultdict(list))  # pipe -> 指标 -> [值]
    for f in files:
        r = analyze_trace(f, args.threshold, args.top, args.min_activity)
        if r is None:
            print(f"[警告] {f}: 无有效 X 事件，跳过", file=sys.stderr)
            continue
        all_results.append(r)
        if len(files) == 1 or len(files) <= 3:
            sub = "self" if not r["merged_schema"] else None
            if sub:
                fmt_subcore("(单文件)", r["subcores"][sub], r["span"], args.threshold, None,
                            args.min_activity, r.get("t0"), r.get("t1"), r.get("waits"))
            else:
                for sc, pipes in r["subcores"].items():
                    fmt_subcore(sc, pipes, r["span"], args.threshold, None, args.min_activity,
                                r.get("t0"), r.get("t1"), r.get("waits"))
            print(f"-- WAIT_FLAG 归因（top {args.top}，按等待时长降序）--")
            for wpipe, trig, flag, cnt, dur, pct in r["waits"][: args.top]:
                flag_s = f" FLAG:{flag}" if flag not in (None, "?") else ""
                print(f"  {wpipe} ← {trig}{flag_s:<9} x{cnt:<6} dur={dur:10.2f} ({pct:5.1f}% span)")
        for pr in r["pairs"]:
            pair_collect[frozenset((pr["pipe_a"], pr["pipe_b"]))].append(pr["overlap_pct"])
        for w in r["waits"]:
            wait_collect[(w[0], w[1], w[2])][0] += w[3]
            wait_collect[(w[0], w[1], w[2])][1] += w[4]
        for pipes in r["subcores"].values():
            for p, s in pipes.items():
                pipe_collect[p]["merged_pct"].append(s["merged_pct"])
                pipe_collect[p]["wait_pct"].append(s["wait_pct"])
                pipe_collect[p]["bar_pct"].append(s["bar_pct"])
                pipe_collect[p]["eff_pct"].append(s["eff_pct"])

    if not all_results:
        print("[错误] 没有任何有效分析结果", file=sys.stderr)
        sys.exit(2)

    # 跨文件/子核汇总（中位数）
    print(f"\n===== 汇总（{len(all_results)} 个 trace 文件，重叠度=交集/min(活跃)，取中位数 [min~max]）=====")
    print("-- pipe 活跃四口径（中位数：merged=队列占用 / wait=空等 / bar=汇合同步 / eff=有效计算）--")
    for p, mm in sorted(pipe_collect.items(), key=lambda kv: -sorted(kv[1]["merged_pct"])[len(kv[1]["merged_pct"]) // 2]):
        def med(vals):
            v = sorted(vals)
            return v[len(v) // 2]
        print(f"  {p:<12} merged={med(mm['merged_pct']):6.1f}%  wait={med(mm['wait_pct']):6.1f}%  "
              f"bar={med(mm['bar_pct']):5.1f}%  eff={med(mm['eff_pct']):6.1f}%")
    for pair, vals in sorted(pair_collect.items(), key=lambda kv: -sorted(kv[1])[len(kv[1]) // 2]):
        vals_s = sorted(vals)
        med = vals_s[len(vals_s) // 2]
        pa, pb = sorted(pair)
        tag = PPT_PAIRS.get(pair, "")
        verdict = "★掩盖失效" if med < args.threshold else "掩盖成立"
        print(f"  {pa:<10} x {pb:<10} 中位 {med:>6.2f}%  [{min(vals_s):.2f}%~{max(vals_s):.2f}%]  "
              f"{verdict}  {('['+tag+']') if tag else ''}")
    print(f"-- WAIT_FLAG 归因汇总（top {args.top}）--")
    rows = sorted(wait_collect.items(), key=lambda kv: -kv[1][1])
    for (wpipe, trig, flag), (cnt, dur) in rows[: args.top]:
        flag_s = f" FLAG:{flag}" if flag not in (None, "?") else ""
        print(f"  {wpipe} ← {trig}{flag_s:<9} x{cnt:<6} dur={dur:12.2f}")

    if args.json:
        slim = []
        for r in all_results:
            slim.append({
                "file": r["file"], "span": r["span"], "merged_schema": r["merged_schema"],
                "subcores": {sc: {p: {k: v for k, v in s.items()
                                      if k not in ("merged", "wait_merged", "bar_merged", "eff_merged")}
                                   for p, s in pipes.items()}
                              for sc, pipes in r["subcores"].items()},
                "pairs": r["pairs"], "waits": r["waits"],
            })
        with open(args.json, "w") as f:
            json.dump({"threshold": args.threshold, "results": slim}, f, ensure_ascii=False, indent=1)
        print(f"\n[JSON] 已写入 {args.json}")


if __name__ == "__main__":
    main()
