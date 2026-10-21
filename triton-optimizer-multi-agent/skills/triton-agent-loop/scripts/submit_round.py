#!/usr/bin/env python3
"""
Submit Round Validation Script

Validates a completed optimization round and provides structured guidance
for the next step (continue, stop, or switch direction).

Usage:
    python3 submit_round.py \
        --round-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag}/opt-round-{N} \
        --current-round {N} \
        --final-round {max_rounds} \
        [--op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag}]

Output (JSON):
    {
        "status": "pass" | "fail",
        "issues": [...],         # 产物完整性校验失败项
        "warnings": [...],       # 非致命警告
        "guideline": "...",      # 仅供参考，不具决策权
        "local_optimum": { ... } # 局部最优检测结果，供方向去重参考
    }

⚠️ 本脚本仅做合同校验（产物完整性检查），不再决定停止/继续流向。
Agent 必须执行 §2.2 Step 1→2→3 自主判定，不受本脚本输出的 next_option/approved_stop 影响。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from check_local_optimum import check_local_optimum
from strategies import load_strategy_for_algorithm
from config_loader import read_config


def _is_worker_mode() -> bool:
    """Return True if running as a per-round sub-agent worker."""
    return os.environ.get("TRITON_OPTIMIZER_MODE", "").lower() == "worker"

_REQUIRED_SUMMARY_FIELDS = [
    "success", "round_index", "round_strategy",
    "hypothesis", "direction",
    "gen_iterations",
]

_REQUIRED_SUMMARY_FIELDS_OPTIMIZED = [
    "optimized", "best_speedup", "target_speedup",
    "target_reached", "perf_data",
]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


def find_op_name(work_dir: Path | str) -> str | None:
    """Try to find op_name from summary.json or generated code files."""
    if isinstance(work_dir, str):
        work_dir = Path(work_dir)
    summary_path = work_dir / "summary.json"
    if summary_path.is_file():
        try:
            data = load_json(summary_path)
            if "op_name" in data:
                return str(data["op_name"])
        except ValueError:
            pass

    # Scan for *generated.py files
    for path in sorted(work_dir.glob("*_generated.py")):
        return path.stem.replace("_generated", "")

    # Scan for .py files (excluding known metadata)
    for path in sorted(work_dir.glob("*.py")):
        if path.name not in ("round_contract_check.py", "check_local_optimum.py"):
            return path.stem
    return None


def check_round_is_idle(round_dir: Path | str, op_dir: Path | None) -> bool:
    """Detect if the current round is idle (no semantic code change vs best baseline)."""
    if isinstance(round_dir, str):
        round_dir = Path(round_dir)
    if op_dir is None:
        return False
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)
    if not op_dir.is_dir():
        return False

    op_name = find_op_name(round_dir)
    if op_name is None:
        return False

    generated = round_dir / f"{op_name}_generated.py"
    if not generated.is_file():
        return False

    # Find the best historical round (highest best_speedup, excluding self)
    round_index_path = op_dir / "round_index.json"
    if not round_index_path.is_file():
        return False

    try:
        round_index = load_json(round_index_path)
    except ValueError:
        return False

    if not isinstance(round_index, list):
        return False

    current_round_name = round_dir.name
    best_entry = None
    for entry in round_index:
        if entry.get("round_dir") == current_round_name:
            continue
        if entry.get("status") == "success":
            speedup = entry.get("best_speedup", 0)
            if isinstance(speedup, (int, float)) and speedup > 0:
                if best_entry is None or speedup > best_entry.get("best_speedup", 0):
                    best_entry = entry

    if best_entry is None:
        return False

    best_dir = op_dir / best_entry["round_dir"]
    best_code = best_dir / f"{op_name}_generated.py"
    if not best_code.is_file():
        return False

    # Compare file content (byte-level)
    return generated.read_bytes() == best_code.read_bytes()


def compute_round_metadata(
    round_dir: Path,
    op_dir: Path | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute direction_yield, yield_reason, absolute_gain for a round.

    Reads summary.json and compares with the baseline recorded in
    baseline_dir to compute gain. Uses the strategy configured for the
    operator (defaulting to naive).
    """
    result: dict[str, Any] = {
        "direction_yield": None,
        "yield_reason": None,
        "absolute_gain_vs_baseline": None,
        "absolute_gain_vs_global_baseline": None,
        "baseline_dir": None,
        "global_baseline_dir": None,
    }
    summary_path = round_dir / "summary.json"
    if not summary_path.is_file():
        return result
    try:
        summary = load_json(summary_path)
    except ValueError:
        return result

    # Load the strategy for the algorithm recorded in summary.json, supplied by
    # the caller context, or stored in the orchestrator state file, defaulting to
    # naive only when no algorithm is available.
    strategy = None
    algorithm = context.get("algorithm") if context else None
    if not isinstance(algorithm, str):
        algorithm = summary.get("algorithm")
    if not isinstance(algorithm, str) and op_dir is not None:
        plugin_root = op_dir.parent.parent
        state_path = plugin_root / ".triton-agent" / f"state-{op_dir.name}.json"
        if state_path.is_file():
            try:
                algorithm = load_json(state_path).get("algorithm")
            except ValueError:
                pass
    if not isinstance(algorithm, str):
        algorithm = "naive"
    if op_dir is not None:
        plugin_root = op_dir.parent.parent
        config_path = plugin_root / "config.json"
        if config_path.is_file():
            try:
                config = load_json(config_path)
                strategy = load_strategy_for_algorithm(config, algorithm)
            except (ValueError, Exception):
                strategy = None
    if strategy is None:
        strategy = load_strategy_for_algorithm(
            {"algorithms": {"naive": {"enabled": True, "params": {}}}}, "naive"
        )

    best_speedup = summary.get("best_speedup", 0.0)
    target_speedup = summary.get("target_speedup", 5.0)
    baseline_speedup = 0.0
    baseline_dir_name = summary.get("baseline_dir")
    global_baseline_speedup = 0.0

    if op_dir and baseline_dir_name:
        baseline_summary_path = op_dir / baseline_dir_name / "summary.json"
        if baseline_summary_path.is_file():
            try:
                baseline_summary = load_json(baseline_summary_path)
                bs = baseline_summary.get("best_speedup", 0.0)
                if isinstance(bs, (int, float)):
                    baseline_speedup = bs
                result["baseline_dir"] = baseline_dir_name
            except ValueError:
                pass

    if op_dir:
        global_baseline_info_path = op_dir / "global_baseline" / "baseline_info.json"
        if global_baseline_info_path.is_file():
            try:
                global_baseline_info = load_json(global_baseline_info_path)
                gbs = global_baseline_info.get("baseline_speedup", 0.0)
                if isinstance(gbs, (int, float)):
                    global_baseline_speedup = gbs
                result["global_baseline_dir"] = "global_baseline"
            except ValueError:
                pass
        # If no explicit baseline info, assume the global baseline has speedup 1.0
        # (i.e., it is the reference implementation).
        if global_baseline_speedup <= 0.0:
            global_baseline_speedup = 1.0
            result["global_baseline_dir"] = "global_baseline"

    perf_data = summary.get("perf_data") or {}
    total = perf_data.get("total_cases", 0)
    passed = perf_data.get("passed_cases", 0)
    verify_passed = (
        isinstance(total, (int, float))
        and isinstance(passed, (int, float))
        and int(total) > 0
        and int(passed) == int(total)
    )

    improvement_made = bool(summary.get("optimized", False))
    timeout = summary.get("status") == "timeout"

    if context is None:
        context = {}
    context.setdefault("op_dir", str(op_dir) if op_dir else None)
    context.setdefault("round_dir", str(round_dir))

    yield_label, yield_reason, absolute_gain = strategy.classify_direction_yield(
        best_speedup=best_speedup if isinstance(best_speedup, (int, float)) else 0.0,
        baseline_speedup=baseline_speedup,
        target_speedup=target_speedup if isinstance(target_speedup, (int, float)) else 5.0,
        verify_passed=verify_passed,
        improvement_made=improvement_made,
        timeout=timeout,
        idle=False,
        context=context,
        global_baseline_speedup=global_baseline_speedup,
    )
    result["direction_yield"] = yield_label
    result["yield_reason"] = yield_reason
    result["absolute_gain_vs_baseline"] = round(absolute_gain, 6)

    absolute_gain_global = (
        best_speedup - global_baseline_speedup
        if isinstance(best_speedup, (int, float)) and global_baseline_speedup > 0
        else 0.0
    )
    result["absolute_gain_vs_global_baseline"] = round(absolute_gain_global, 6)
    return result


def validate_round_directory(round_dir: Path) -> list[str]:
    """Validate required artifacts exist in the round directory."""
    if isinstance(round_dir, str):
        round_dir = Path(round_dir)
    issues = []
    op_name = find_op_name(round_dir)
    if op_name is None:
        issues.append("cannot determine operator name from round directory")
        return issues

    # Check summary.json
    summary_path = round_dir / "summary.json"
    if not summary_path.is_file():
        issues.append("missing summary.json")
        return issues

    # Check generated code
    generated_path = round_dir / f"{op_name}_generated.py"
    if not generated_path.is_file():
        output_gen = round_dir / "output" / "generated_code.py"
        output_opt = round_dir / "output" / "optimized_code.py"
        if output_opt.is_file():
            if output_opt.stat().st_size <= 100:
                issues.append(f"{op_name}_generated.py not found and output/optimized_code.py is too small")
        elif output_gen.is_file():
            if output_gen.stat().st_size <= 100:
                issues.append(f"{op_name}_generated.py not found and output/generated_code.py is too small")
        else:
            issues.append(f"missing output kernel code (neither {op_name}_generated.py nor output/optimized_code.py found)")

    # Check verify result (only Phase 3 iter_* results)
    verify_results = sorted(round_dir.glob("output/iter_*/verify/verify_result.json"))
    if not verify_results:
        issues.append("missing verify_result.json in output/iter_*/verify/")

    # Check Phase 4 artifacts (only when phase4_entered is true)
    try:
        summary = load_json(summary_path) if summary_path.is_file() else {}
    except ValueError:
        summary = {}
    if summary.get("phase4_entered", False):
        opt_iter_dirs = sorted(round_dir.glob("output/opt_iter_*"))
        if not opt_iter_dirs:
            issues.append("phase4_entered=true but no opt_iter_*/ directory found in output/")
        else:
            latest_opt = opt_iter_dirs[-1]
            opt_verify = latest_opt / "verify" / "verify_result_optimized.json"
            if not opt_verify.is_file():
                issues.append(f"phase4_entered=true but {latest_opt.name}/verify/verify_result_optimized.json not found")

            # Verify at least one opt_iter has code differing from generated_code
            # (prevent identity-copy "optimizations" that bypass Phase 4)
            generated_code = round_dir / "output" / "generated_code.py"
            has_real_change = False
            for od in opt_iter_dirs:
                opt_code = od / "optimized_code.py"
                if opt_code.is_file() and generated_code.is_file():
                    if opt_code.read_bytes() != generated_code.read_bytes():
                        has_real_change = True
                        break
            if not has_real_change and opt_iter_dirs:
                issues.append(
                    "None of the opt_iter_*/optimized_code.py files differ from "
                    "output/generated_code.py. All optimization iterations produced "
                    "byte-identical code — no actual optimization was performed. "
                    "At least one opt_iter must apply a real code change."
                )

        attempts_path = round_dir / "output" / "attempts.md"
        if not attempts_path.is_file():
            issues.append("phase4_entered=true but output/attempts.md not found")
        else:
            lines = attempts_path.read_text().splitlines()
            record_count = sum(1 for line in lines
                              if line.strip().startswith("- [") and ("未命中" in line or "已尝试" in line or "已命中" in line))
            if record_count < 26:
                issues.append(
                    f"output/attempts.md has only {record_count} optimization point records (need >= 26)"
                )
            # authenticity check: at least 1 point must be actually attempted
            tried_count = sum(1 for line in lines
                              if line.strip().startswith("- [") and ("已尝试" in line or "已命中" in line))
            if tried_count < 1:
                issues.append(
                    "output/attempts.md has no '已尝试' or '已命中' records "
                    f"(found {tried_count}). At least 1 optimization point must be "
                    "actually attempted — run verify + benchmark — even if the model "
                    "believes no optimization point applies. Select the most reasonable "
                    "point and execute a real optimization iteration."
                )

    return issues


def validate_summary(summary_path: Path) -> list[str]:
    """Validate summary.json field completeness."""
    issues = []
    try:
        data = load_json(summary_path)
    except ValueError as exc:
        issues.append(f"summary.json parse error: {exc}")
        return issues

    for field in _REQUIRED_SUMMARY_FIELDS:
        if field not in data:
            issues.append(f"summary.json missing required field: {field}")

    if data.get("success", False):
        perf_data = data.get("perf_data")
        if not isinstance(perf_data, dict):
            issues.append("summary.json: perf_data missing or not a dict")
        elif "total_cases" not in perf_data or "passed_cases" not in perf_data:
            issues.append("summary.json: perf_data missing total_cases/passed_cases")

    # phase4_entered check: unconditional, NOT gated on success flag.
    # If Phase 3 verify passed on disk, this will also be caught by
    # validate_round Step 3b (disk-based), but this check provides
    # an additional layer regardless of success value.
    if not data.get("phase4_entered"):
        # Only flag if there's evidence Phase 3 was attempted
        gen_iters = data.get("gen_iterations", 0)
        if gen_iters > 0 or "passed_cases" in data.get("perf_data", {}):
            issues.append(
                "summary.json: phase4_entered must be true when Phase 3 was executed. "
                "Phase 4 optimization was skipped."
            )

    return issues


def check_verify_passed(round_dir: Path) -> tuple[bool, str]:
    """Check that verify_result.json has passed_cases == total_cases > 0.
    Only checks Phase 3 iter_* results, not Phase 4 opt_iter_*."""
    verify_paths = sorted(round_dir.glob("output/iter_*/verify/verify_result.json"))
    if not verify_paths:
        return False, "no verify_result.json found"

    # Use the latest verify result
    target = verify_paths[-1]
    try:
        data = load_json(target)
    except ValueError as exc:
        return False, f"verify_result.json parse error: {exc}"

    total = data.get("total_cases", 0)
    passed = data.get("passed_cases", 0)
    if not isinstance(total, (int, float)) or not isinstance(passed, (int, float)):
        return False, f"verify_result.json: non-numeric total_cases({total})/passed_cases({passed})"

    total = int(total)
    passed = int(passed)
    if total == 0:
        return False, "verify_result.json: total_cases is 0"
    if passed != total:
        failures = data.get("failures", [])
        fail_summary = ", ".join(f.get("error_type", "unknown") for f in failures[:3])
        return False, f"verify_result.json: passed({passed}) < total({total}), failures: {fail_summary}"
    return True, "passed"


def validate_round(
    round_dir: Path,
    current_round: int | None = None,
    final_round: int | None = None,
    op_dir: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "pass",
        "issues": [],
        "warnings": [],
        "guideline": "",
        "suggested_next_dir": None,
        "round_type": "active",  # "active" | "idle"
        "stop_decision_notes": "stop/continue decision moved to Agent §2.2 Step 3b (C1-C4)",
        "direction_yield": None,
        "yield_reason": None,
        "absolute_gain_vs_baseline": None,
        "absolute_gain_vs_global_baseline": None,
        "baseline_dir": None,
        "global_baseline_dir": None,
    }

    # 1. Validate directory structure
    dir_issues = validate_round_directory(round_dir)
    result["issues"].extend(dir_issues)

    # 2. Validate summary.json
    summary_path = round_dir / "summary.json"
    if summary_path.is_file():
        summary_issues = validate_summary(summary_path)
        result["issues"].extend(summary_issues)

    # 3. Check verify passed
    verify_ok, verify_msg = check_verify_passed(round_dir)
    if not verify_ok:
        result["issues"].append(f"verify check failed: {verify_msg}")

    # 3b. Unconditional Phase 4 check (not gated on success flag)
    # If Phase 3 verification passed, Phase 4 MUST have been executed.
    # This cannot be bypassed by setting success=false in summary.json.
    if verify_ok:
        summary_path = round_dir / "summary.json"
        if summary_path.is_file():
            try:
                summary_data = load_json(summary_path)
                if not summary_data.get("phase4_entered", False):
                    result["issues"].append(
                        "Phase 3 verification passed (all cases) but phase4_entered is false. "
                        "Phase 4 optimization was skipped. "
                        "Every round with verified Phase 3 output MUST also execute Phase 4 "
                        "optimization before transition. Return to Phase 4 and complete at "
                        "least one optimization iteration."
                    )
            except ValueError:
                pass

    # 3c. Check plateau_review round integrity
    # plateau_review must NOT have Phase 4 artifacts (opt_iter_*, phase4_entered=true)
    # and MUST have analysis.md
    summary_path = round_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary_data = load_json(summary_path)
            round_strategy = summary_data.get("round_strategy", "")
            if round_strategy == "plateau_review":
                # Must have analysis.md
                analysis_path = round_dir / "analysis.md"
                if not analysis_path.is_file():
                    result["issues"].append(
                        "plateau_review round missing analysis.md — "
                        "must produce analysis report before transitioning"
                    )
                # Must NOT have Phase 4 artifacts
                if summary_data.get("phase4_entered", False):
                    result["issues"].append(
                        "plateau_review round has phase4_entered=true but should be "
                        "analysis-only. Phase 4 optimization artifacts are not expected "
                        "in a plateau_review round."
                    )
                opt_dirs = sorted(round_dir.glob("output/opt_iter_*"))
                if opt_dirs:
                    result["issues"].append(
                        f"plateau_review round has {len(opt_dirs)} opt_iter_*/ "
                        f"directories but should not have Phase 4 artifacts. "
                        f"plateau_review is analysis-only, no code changes allowed."
                    )
        except ValueError:
            pass

    # 4. Run local optimum detection (orchestrator mode only)
    if not _is_worker_mode() and op_dir and op_dir.is_dir():
        try:
            local_opt = check_local_optimum(op_dir)
            result["local_optimum"] = local_opt
            if local_opt.get("warnings"):
                result["warnings"].extend(local_opt["warnings"])
        except Exception as exc:
            result["warnings"].append(f"local_optimum check error: {exc}")

    # 5. Idle detection (orchestrator mode only)
    if not _is_worker_mode():
        round_idle = check_round_is_idle(round_dir, op_dir)
        if round_idle:
            result["round_type"] = "idle"
            result["warnings"].append(
                f"Round {current_round} is idle: generated code is identical "
                "to the historical best baseline. No semantic changes made."
            )

    # 5.5 Compute direction-level causal metadata.
    if not _is_worker_mode():
        metadata = compute_round_metadata(
            round_dir,
            op_dir,
            context={"op_dir": str(op_dir) if op_dir else None, "round_dir": str(round_dir)},
        )
        result.update(metadata)

    # 6. [DEPRECATED] Stop/continue evaluation: moved to Agent §2.2 Step 3b
    # submit_round.py no longer decides stop/continue.
    # C1/C2/C3/D4 are evaluated by the Agent in §2.2 Step 3b.

    # 7. Determine status and suggested_next_dir (informational only)
    if result["issues"]:
        result["status"] = "fail"
        result["guideline"] = (
            f"Round {current_round} has {len(result['issues'])} unresolved issue(s). "
            "Fix them before continuing. Issues: " + "; ".join(result["issues"])
        )
    else:
        result["guideline"] = (
            f"Round {current_round} passed validation. "
            "Proceed to §2.2 Step 1→2→3 for stop/continue decision."
        )
        if current_round is not None:
            result["suggested_next_dir"] = f"opt-round-{current_round + 1}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a completed optimization round"
    )
    parser.add_argument("--round-dir", required=True, help="Round directory path")
    parser.add_argument("--current-round", type=int, default=None, help="Current round number")
    parser.add_argument("--final-round", type=int, default=None, help="Final round number (max_rounds)")
    parser.add_argument("--op-dir", default=None, help="Operator directory path (for local optimum detection)")
    args = parser.parse_args()

    round_dir = Path(args.round_dir).expanduser().resolve()
    if not round_dir.is_dir():
        print(json.dumps({
            "status": "fail",
            "issues": [f"round directory not found: {round_dir}"],
            "warnings": [],
            "guideline": "Create the round directory before running submit-round.",
            "next_option": None,
        }))
        return 1

    # Auto-detect op_dir if not provided (orchestrator mode only)
    op_dir = None
    if not _is_worker_mode():
        if args.op_dir:
            op_dir = Path(args.op_dir).expanduser().resolve()
        else:
            # op_dir is the parent of round_dir (e.g., triton_ascend_output/{op_name})
            parent = round_dir.parent
            if (parent / "round_index.json").is_file():
                op_dir = parent

    result = validate_round(round_dir, args.current_round, args.final_round, op_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
