#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""collect_from_workdir.py — 从 triton-op-generator 工作目录收集回流原料，生成 reflux_manifest.json 骨架。

triton-knowledge-reflux skill 的 P1 采集器。只做确定性收集与触发判定（零 LLM），
蒸馏内容（候选知识条目、案例卡正文）由 Agent 按 SKILL.md 流程填入 `agent_fill` 区。

收集来源（全部只读）：
  summary.json                     顶层成败/迭代/加速比
  report.md                        架构说明、优化历程（仅探测 arch 行）
  precheck.json                    category / template / Layer 1 约束
  output/iter_*/verify/verify_result.json        Phase 3 逐轮精度（含 failures 明细）
  output/iter_*/perf_result.json                 Phase 3 逐轮性能
  output/opt_iter_*/(optimized_perf_result.json,
                     baseline_perf_result.json,
                     verify/verify_result_optimized.json)   Phase 4 逐轮优化
  knowledge/b_context_*.json                     本次消费的知识库卡片（S4 反馈信号）

Usage:
  collect_from_workdir.py --work-dir <工作目录> [--output <manifest 路径>] [--max-failures N]
  collect_from_workdir.py --work-dir <工作目录> --check-only   # 只打印触发判定，不落盘

Exit code: 0 正常（含"全部未触发"）；2 工作目录/summary.json 缺失或不可解析。
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

SCHEMA = "reflux_manifest.v1"

# S1 阈值：成功案例的几何平均加速比下限（与 AGENTS.md Phase 6 Source B 对齐）
S1_MIN_SPEEDUP = 1.1

FAILURE_MSG_TRUNC = 300


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _warn(warnings, msg):
    warnings.append(msg)


# ---------------------------------------------------------------- summary ----

def collect_run_facts(work_dir, warnings):
    summary = _load_json(os.path.join(work_dir, "summary.json"))
    if summary is None:
        _warn(warnings, "summary.json 缺失或不可解析")
        return None
    perf = summary.get("perf_data") or {}
    return {
        "success": summary.get("success"),
        "gen_iterations": summary.get("gen_iterations"),
        "opt_iterations": summary.get("opt_iterations"),
        "optimized": summary.get("optimized"),
        "target_speedup": summary.get("target_speedup"),
        "target_reached": summary.get("target_reached"),
        "best_speedup": summary.get("best_speedup"),
        "failure_phase": summary.get("failure_phase"),
        "failure_reason": summary.get("failure_reason"),
        "gpu_mode": summary.get("gpu_mode", False),
        "framework_latency_source": summary.get("framework_latency_source"),
        "perf": {
            "speedup_vs_torch": perf.get("speedup_vs_torch"),
            "avg_latency_ms": perf.get("avg_latency_ms"),
            "total_cases": perf.get("total_cases"),
            "passed_cases": perf.get("passed_cases"),
            "failed_cases": perf.get("failed_cases"),
            "anomaly_indices": {
                k: perf.get(k) for k in
                ("nan_indices", "inf_indices", "zero_indices", "negative_indices", "none_indices")
                if perf.get(k)
            },
            "evidence": "summary.json#perf_data",
        },
        "evidence": "summary.json",
    }


def _parse_op_name(work_dir):
    """目录名 op_{op_index}_{op_name}_{YYYYMMDD_HHMM}_{rid} → 提取 op_name（含 index 语义时保留原样）。"""
    base = os.path.basename(work_dir.rstrip("/"))
    m = re.match(r"op_(\d+)_(.+?)_(\d{8}_\d{4})_\d{4}$", base)
    if m:
        return m.group(2)
    return base


def detect_arch(report_path, warnings):
    """从 report.md 的「**arch**: ascend910b1」行探测硬件架构。"""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read(20000)
    except OSError:
        _warn(warnings, "report.md 不可读，arch 留空")
        return None
    m = re.search(r"\*\*arch\*\*[:：]\s*(\S+)", text)
    return m.group(1).rstrip("| ") if m else None


def collect_precheck(work_dir, warnings):
    pre = _load_json(os.path.join(work_dir, "precheck.json"))
    if pre is None:
        _warn(warnings, "precheck.json 缺失（Phase 2 Step 1 门禁产物应必在）")
        return None
    keys = ("category", "template_path", "template_exists", "writing_class",
            "index_path", "loaded_via", "new_category")
    out = {k: pre.get(k) for k in keys if pre.get(k) is not None}
    l1 = pre.get("layer1_constraints_loaded")
    if isinstance(l1, list):
        out["layer1_constraint_count"] = len(l1)
    return out


# ------------------------------------------------------------- phase 3/4 ----

def _digest_failures(verify, max_failures):
    out = []
    for fail in (verify.get("failures") or [])[:max_failures]:
        out.append({
            "case_idx": fail.get("case_idx"),
            "input_desc": (fail.get("input_desc") or "")[:120] or None,
            "error_type": fail.get("error_type"),
            "error_msg": (fail.get("error_msg") or "")[:FAILURE_MSG_TRUNC],
        })
    if verify.get("failures") and len(verify.get("failures")) > max_failures:
        out.append({"note": "failures 截断，全量见 verify_result.json"})
    return out


def collect_phase3(work_dir, max_failures, warnings):
    iters = []
    for d in sorted(glob.glob(os.path.join(work_dir, "output", "iter_*"))):
        if not os.path.isdir(d):
            continue
        rel = os.path.relpath(d, work_dir)
        verify = _load_json(os.path.join(d, "verify", "verify_result.json"))
        perf = _load_json(os.path.join(d, "perf_result.json"))
        if verify is None:
            _warn(warnings, "%s: verify_result.json 缺失" % rel)
        iters.append({
            "dir": rel,
            "log": os.path.join(rel, "log.md") if os.path.exists(os.path.join(d, "log.md")) else None,
            "verify": None if verify is None else {
                "total_cases": verify.get("total_cases"),
                "passed_cases": verify.get("passed_cases"),
                "all_passed": verify.get("passed_cases") == verify.get("total_cases")
                              and (verify.get("total_cases") or 0) > 0,
                "failures_digest": _digest_failures(verify, max_failures),
                "evidence": os.path.join(rel, "verify", "verify_result.json"),
            },
            "perf": None if perf is None else {
                "speedup_vs_torch": perf.get("speedup_vs_torch"),
                "evidence": os.path.join(rel, "perf_result.json"),
            },
        })
    return iters


def collect_phase4(work_dir, warnings):
    records = []
    for d in sorted(glob.glob(os.path.join(work_dir, "output", "opt_iter_*"))):
        if not os.path.isdir(d):
            continue
        rel = os.path.relpath(d, work_dir)
        opt_perf = _load_json(os.path.join(d, "optimized_perf_result.json"))
        base_perf = _load_json(os.path.join(d, "baseline_perf_result.json"))
        verify = _load_json(os.path.join(d, "verify", "verify_result_optimized.json"))
        # 轮次命名：opt_iter_{n} / opt_iter_{n}_ir_{k} / opt_iter_{n}_bs_{BLOCK}
        m = re.match(r"opt_iter_(\d+)(?:_(ir|bs)_(\w+))?$", os.path.basename(d))
        kind = m.group(2) if m else None
        if opt_perf is None:
            _warn(warnings, "%s: optimized_perf_result.json 缺失（本轮可能被 L1 拒绝或未跑 benchmark）" % rel)
        base_speedup = base_perf.get("speedup_vs_torch") if base_perf else None
        opt_speedup = opt_perf.get("speedup_vs_torch") if opt_perf else None
        verdict = None
        if opt_speedup is not None and base_speedup is not None:
            verdict = "improved" if opt_speedup > base_speedup else "no_improvement"
        abs_log = os.path.join(d, "log.md")
        log_path = os.path.join(rel, "log.md")
        hit_point = None
        if os.path.exists(abs_log):
            try:
                with open(abs_log, "r", encoding="utf-8", errors="replace") as f:
                    mhit = re.search(r"(?:命中点|hit_optimization_point)\s*[:：]?\s*(\d{1,2})",
                                     f.read(8000))
                hit_point = int(mhit.group(1)) if mhit else None
            except OSError:
                pass
        records.append({
            "dir": rel,
            "round_kind": kind or "normal",   # normal | ir | bs | simulator 轮由 log 区分
            "hit_optimization_point": hit_point,   # 自动提取，未命中留 Agent 补
            "log": log_path if os.path.exists(abs_log) else None,
            "verify_optimized": None if verify is None else {
                "total_cases": verify.get("total_cases"),
                "passed_cases": verify.get("passed_cases"),
                "all_passed": verify.get("passed_cases") == verify.get("total_cases")
                              and (verify.get("total_cases") or 0) > 0,
                "failures_digest": _digest_failures(verify, 3),
                "evidence": os.path.join(rel, "verify", "verify_result_optimized.json"),
            },
            "speedup": {
                "baseline": base_speedup,
                "optimized": opt_speedup,
                "delta": (opt_speedup - base_speedup)
                         if (opt_speedup is not None and base_speedup is not None) else None,
                "evidence": os.path.join(rel, "optimized_perf_result.json")
                            + " vs " + os.path.join(rel, "baseline_perf_result.json"),
            },
            "verdict_hint": verdict,   # improved / no_improvement；无提升同样是知识（S3）
        })
    return records


# --------------------------------------------------- S2 修复闭环 / S4 消费 ----

def detect_debug_loops(phase3, phase4):
    """S2：存在「失败 → 后续某轮全过」的修复闭环（Phase 3 迭代对 + Phase 4 验证失败回退）。"""
    loops = []
    verify_seq = []
    for it in phase3:
        if it["verify"]:
            verify_seq.append(("phase3/" + it["dir"], it["verify"]))
    for rec in phase4:
        if rec["verify_optimized"]:
            verify_seq.append(("phase4/" + rec["dir"], rec["verify_optimized"]))
    failed = [(name, v) for name, v in verify_seq if not v["all_passed"]]
    passed = [name for name, v in verify_seq if v["all_passed"]]
    for name, v in failed:
        later_pass = [p for p in passed if p > name]   # 字典序 = 目录序，同前缀下成立
        if later_pass:
            loops.append({
                "failed_at": name,
                "failures_digest": v["failures_digest"],
                "resolved_at": later_pass[0],
            })
    return loops


def collect_consumed_cards(work_dir, warnings):
    """S4：解析 knowledge/b_context_*.json，列出本次消费的知识库卡。

    生成侧常用 `>` 重定向保存检索结果，stdout 里可能混入 WARN 等前缀行，
    这里跳过首个 '{' 之前的字节做容错解析。
    """
    consumed = []
    for path in sorted(glob.glob(os.path.join(work_dir, "knowledge", "b_context_*.json"))):
        rel = os.path.relpath(path, work_dir)
        m = re.match(r"b_context_(design|optimize|repair)(?:_(\d+))?_(\w+)\.json$", os.path.basename(path))
        phase = m.group(1) if m else os.path.basename(path)
        sub = m.group(3) if m else ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            data = json.loads(raw[raw.index("{"):])
        except (OSError, ValueError):
            _warn(warnings, "%s: 不可解析" % rel)
            continue
        for r in (data.get("results") or []):
            consumed.append({
                "doc_id": r.get("doc_id") or r.get("path"),
                "title": r.get("title"),
                "kind": r.get("kind"),
                "relevance_level": (r.get("relevance") or {}).get("level"),
                "recommended_action": (r.get("relevance") or {}).get("recommended_action"),
                "phase": phase,          # design / optimize / repair
                "lane": sub,             # preflight / runbooks / ops
                "context_file": rel,
            })
    return consumed


# ------------------------------------------------------------- 触发判定 ----

def evaluate_triggers(run_facts, phase3, phase4, debug_loops, consumed):
    perf = (run_facts or {}).get("perf") or {}
    passed = perf.get("passed_cases")
    total = perf.get("total_cases")
    speedup = perf.get("speedup_vs_torch")

    s1_ok = bool(run_facts and run_facts.get("success")
                 and passed == total and (total or 0) > 0
                 and isinstance(speedup, (int, float)) and speedup >= S1_MIN_SPEEDUP)
    s1 = {
        "triggered": s1_ok,
        "reason": ("success + %s/%s 全过 + geomean %s >= %s"
                   % (passed, total, speedup, S1_MIN_SPEEDUP)) if s1_ok
                  else "不满足 success / 精度全过 / speedup_vs_torch ≥ %s 之一" % S1_MIN_SPEEDUP,
    }

    s2 = {
        "triggered": bool(debug_loops),
        "resolved_pairs": debug_loops,
        "reason": ("%d 组「错误→修复→通过」闭环" % len(debug_loops)) if debug_loops
                  else "无 A 类错误修复闭环（或一次通过 / 失败未解决）",
    }

    s3_records = [r for r in phase4 if r["verdict_hint"] is not None
                  or r["hit_optimization_point"] is not None]
    # 优化轮本身即性能洞察原料（含 no_improvement / verify 失败回退）
    s3 = {
        "triggered": bool(phase4),
        "records_hint": s3_records,
        "reason": ("%d 个 Phase 4 优化轮（含无提升/禁用记录，均为 OPT/AP/CT 原料）" % len(phase4))
                  if phase4 else "无 Phase 4 优化轮",
    }

    useful = [c for c in consumed
              if c.get("recommended_action") in ("read",) or c.get("relevance_level") == "strong"]
    s4 = {
        "triggered": bool(useful),
        "consumed_cards": useful,
        "reason": ("%d 张被消费卡（strong 命中）待回填「已知实例」/使用反馈" % len(useful))
                  if useful else "无 strong 命中的知识消费记录",
    }
    return {"S1_success_case": s1, "S2_debug_loop": s2,
            "S3_perf_insight": s3, "S4_consumption_feedback": s4}


# ------------------------------------------------------------------ main ----

def collect(work_dir, max_failures):
    warnings = []
    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "collector": "collect_from_workdir.py",
        "work_dir": os.path.abspath(work_dir),
        "op": {
            "name": _parse_op_name(work_dir),
            "arch": detect_arch(os.path.join(work_dir, "report.md"), warnings),
            "category": None,       # 由 precheck.json 回填，见下方 op.category_assign
        },
        "run_facts": None,
        "precheck": collect_precheck(work_dir, warnings),
        "triggers": {},
        "phase3_iterations": [],
        "phase4_iterations": [],
        "consumed_cards": [],
        "artifacts": {
            "report": "report.md",
            "summary": "summary.json",
            "sketch": "sketch.txt" if os.path.exists(os.path.join(work_dir, "sketch.txt")) else None,
            "final_code": "output/optimized_code.py"
                          if os.path.exists(os.path.join(work_dir, "output", "optimized_code.py"))
                          else ("output/generated_code.py"
                                if os.path.exists(os.path.join(work_dir, "output", "generated_code.py"))
                                else None),
            "session_md": "session.md" if os.path.exists(os.path.join(work_dir, "session.md")) else None,
        },
        "warnings": warnings,
        # ---- 以下为 Agent 蒸馏区（collector 不填）----
        "agent_fill": {
            "candidates": [],
            # 每条候选：{id, type(OPT/AP/CT/DG), dimension, title,
            #   diagnosis(+出处), modification(+出处), verification(+json 字段路径出处),
            #   generalization(Triton通用|条件性+失效边界), placeholders, bad_practice,
            #   route: {target: "ops|runbooks/<category>", existing_entry: <路径或 null>}}
            "case_card": {"route": "ops/<op-name>.md", "outline_done": False},
        },
    }
    if manifest["precheck"] and manifest["precheck"].get("category"):
        manifest["op"]["category"] = manifest["precheck"]["category"]

    summary = _load_json(os.path.join(work_dir, "summary.json"))
    if summary is None:
        return None, warnings
    manifest["run_facts"] = collect_run_facts(work_dir, warnings)
    manifest["phase3_iterations"] = collect_phase3(work_dir, max_failures, warnings)
    manifest["phase4_iterations"] = collect_phase4(work_dir, warnings)
    manifest["consumed_cards"] = collect_consumed_cards(work_dir, warnings)
    debug_loops = detect_debug_loops(manifest["phase3_iterations"], manifest["phase4_iterations"])
    manifest["triggers"] = evaluate_triggers(
        manifest["run_facts"], manifest["phase3_iterations"],
        manifest["phase4_iterations"], debug_loops, manifest["consumed_cards"])
    return manifest, warnings


def main():
    ap = argparse.ArgumentParser(description="收集 op-generator 工作目录 → reflux_manifest.json 骨架")
    ap.add_argument("--work-dir", required=True, help="triton_ascend_output/op_* 工作目录")
    ap.add_argument("--output", default=None, help="manifest 输出路径（默认 <work-dir>/reflux_manifest.json）")
    ap.add_argument("--max-failures", type=int, default=5, help="每轮 failures 明细最多收录条数")
    ap.add_argument("--check-only", action="store_true", help="只打印触发判定摘要，不落盘")
    args = ap.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if not os.path.isdir(work_dir):
        print("ERROR: 工作目录不存在: %s" % work_dir, file=sys.stderr)
        return 2
    if not os.path.exists(os.path.join(work_dir, "summary.json")):
        print("ERROR: summary.json 缺失（任务未完成或目录非 op-generator 工作目录）", file=sys.stderr)
        return 2

    manifest, warnings = collect(work_dir, args.max_failures)
    if manifest is None:
        print("ERROR: 收集失败", file=sys.stderr)
        return 2

    triggers = manifest["triggers"]
    print("== reflux 触发判定（%s）==" % os.path.basename(work_dir))
    any_triggered = False
    for key in ("S1_success_case", "S2_debug_loop", "S3_perf_insight", "S4_consumption_feedback"):
        t = triggers[key]
        any_triggered = any_triggered or t["triggered"]
        print("  %s: %s — %s" % (key, "TRIGGERED" if t["triggered"] else "skip", t["reason"]))
    print("  warnings: %d" % len(manifest["warnings"]))
    for w in manifest["warnings"]:
        print("    [warn] %s" % w)

    if args.check_only:
        return 0
    out_path = os.path.abspath(args.output) if args.output else os.path.join(work_dir, "reflux_manifest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("manifest → %s%s" % (out_path, "" if any_triggered else "（注意：全部未触发，Agent 仍须按 SKILL.md 复核后跳过）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
