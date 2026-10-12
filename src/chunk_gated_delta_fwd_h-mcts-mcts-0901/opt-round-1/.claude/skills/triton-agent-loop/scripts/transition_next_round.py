#!/usr/bin/env python3
"""Transition to Next Round Script

Atomically executes Step 4 (prepare next round) and Phase 0 (directory creation)
of the Agent Loop, ensuring reliable round-to-round transitions.

All rounds are created as normal directories named opt-round-{N+1}. Direction
switch metadata is recorded in round_index.json and opt-note.md, but no special
"-switch" directory is created.

Usage:
    python3 transition_next_round.py \
        --op-dir triton_ascend_output/{op_name} \
        --current-round {N} \
        --next-round-strategy {exploration|structural_change|focused_tuning|stabilization|plateau_review} \
        --next-analysis-policy {pattern_entry|bottleneck_analysis|ir_supported} \
        --next-hypothesis "..." \
        --next-direction "..." \
        --next-evidence-sources '[...]' \
        [--baseline-dir opt-round-K] \
        [--baseline-dirs '["opt-round-K", "opt-round-L"]'] \
        [--switch-marker '{"reason": "...", "previous_direction": "...", "new_direction": "..."}']

Output (JSON):
    {
        "status": "success" | "fail",
        "next_round": <int>,
        "next_round_dir": "opt-round-<N>",
        "baseline_speedup": <float>,
        "issues": [...],
        "warnings": [...]
    }
"""

import argparse
import json
import shutil
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _format_state_template(template: str, op_name: str) -> str | None:
    """Safely format a state file template.

    Only the ``{op_name}`` placeholder is allowed. Returns ``None`` for
    malformed templates instead of raising.
    """
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return None
    fields = {field_name for _, field_name, _, _ in parsed if field_name is not None}
    if fields - {"op_name"}:
        return None
    try:
        return template.format(op_name=op_name)
    except (KeyError, IndexError):
        return None


def _copy_algorithm_state_files(
    source_dir: Path,
    target_dir: Path,
    op_name: str,
    strategy: Any,
) -> list[str]:
    """Copy algorithm-specific state files from source round to next round.

    State file templates are declared by the Strategy class via
    ``state_file_templates``. Each template may contain the ``{op_name}``
    placeholder.

    Returns a list of copied relative paths.
    """
    copied: list[str] = []
    templates = getattr(strategy, "state_file_templates", [])
    for template in templates:
        rel_path = _format_state_template(template, op_name)
        if rel_path is None:
            continue
        src = source_dir / rel_path
        if not src.is_file():
            continue
        dst = target_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel_path)
    return copied


# Import direction selection helpers from the same package.
from direction_selector import (
    build_direction_history,
)
from strategies import discover_strategies, load_strategy_for_algorithm
from strategies.base import SwitchMarker
from config_loader import read_config, get_algorithm_params, get_enabled_algorithms
from workspace_utils import find_op_name


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


def _find_state_dir(op_dir: Path) -> Path | None:
    """查找 .triton-agent/ 目录（从 op_dir 向上搜索）。"""
    for p in [op_dir] + list(op_dir.parents):
        candidate = p / ".triton-agent"
        if candidate.is_dir():
            return candidate
    return None


def _state_path_for_op_dir(op_dir: Path, state_dir_loc: Path | None, op_name: str | None = None) -> Path | None:
    """Return the state file path for the given operator workspace.

    New-format workspaces use ``state-{op_dir.name}.json`` so that different
    algorithms/runs are isolated.  If that file does not exist but a legacy
    ``state-{op_name}.json`` exists, fall back to the legacy path for backward
    compatibility.
    """
    if state_dir_loc is None:
        return None
    new_path = state_dir_loc / f"state-{op_dir.name}.json"
    if new_path.is_file():
        return new_path
    if op_name:
        legacy_path = state_dir_loc / f"state-{op_name}.json"
        if legacy_path.is_file():
            return legacy_path
    return new_path


def _load_round_index(op_dir: Path) -> list[dict[str, Any]]:
    round_index_path = op_dir / "round_index.json"
    if not round_index_path.is_file():
        return []
    try:
        data = load_json(round_index_path)
        if isinstance(data, list):
            return data
    except (ValueError, json.JSONDecodeError):
        pass
    return []


def _entry_for_round_index(round_index: list[dict[str, Any]], round_index_num: int) -> dict[str, Any] | None:
    for entry in round_index:
        if entry.get("round_index") == round_index_num:
            return entry
    return None


def _validate_current_round(current_dir: Path) -> tuple[list[str], dict[str, Any]]:
    """Validate current round and return (issues, current_summary)."""
    issues: list[str] = []
    current_summary: dict[str, Any] = {}

    if not current_dir.is_dir():
        issues.append(f"current round directory not found: {current_dir}")
        return issues, current_summary

    current_summary_path = current_dir / "summary.json"
    if current_summary_path.is_file():
        try:
            current_summary = load_json(current_summary_path)
        except ValueError:
            pass

    # Phase 3 disk verification.
    phase3_verify_paths = sorted(current_dir.glob("output/iter_*/verify/verify_result.json"))
    phase3_ok = False
    if phase3_verify_paths:
        try:
            vdata = load_json(phase3_verify_paths[-1])
            total = vdata.get("total_cases", 0)
            passed = vdata.get("passed_cases", 0)
            if isinstance(total, (int, float)) and isinstance(passed, (int, float)):
                phase3_ok = int(total) > 0 and int(passed) == int(total)
        except (ValueError, json.JSONDecodeError):
            pass

    if not phase3_ok:
        issues.append(
            "Phase 3 verification not found on disk (output/iter_*/verify/verify_result.json)"
        )

    # Phase 4 disk check.
    opt_iter_dirs = sorted(current_dir.glob("output/opt_iter_*"))
    phase4_ok = len(opt_iter_dirs) > 0

    # has_real_change semantic diff.
    has_real_change = False
    generated_code = current_dir / "output" / "generated_code.py"
    if phase4_ok and generated_code.is_file():
        for od in opt_iter_dirs:
            opt_code = od / "optimized_code.py"
            if opt_code.is_file():
                if opt_code.read_bytes() != generated_code.read_bytes():
                    has_real_change = True
                    break
        if not has_real_change:
            issues.append(
                "None of the opt_iter_*/optimized_code.py files differ from "
                "output/generated_code.py. All optimization iterations produced "
                "byte-identical code — no actual optimization was performed."
            )

    # attempts.md authenticity.
    attempts_path = current_dir / "output" / "attempts.md"
    if phase4_ok and not attempts_path.is_file():
        issues.append("phase4_entered but output/attempts.md not found")
    elif attempts_path.is_file():
        lines = attempts_path.read_text().splitlines()
        tried_count = sum(
            1 for line in lines
            if line.strip().startswith("- [") and ("已尝试" in line or "已命中" in line)
        )
        if tried_count < 1 and phase4_ok:
            issues.append(
                "output/attempts.md has no '已尝试' or '已命中' records. "
                "At least 1 optimization point must be actually attempted."
            )

    return issues, current_summary


def _compute_direction_metadata(
    current_summary: dict[str, Any],
    baseline_entry: dict[str, Any] | None,
    strategy: Any,
    plateau_detected: bool = False,
    timeout: bool = False,
    idle: bool = False,
    context: dict[str, Any] | None = None,
    global_baseline_speedup: float = 0.0,
) -> dict[str, Any]:
    """Compute direction_yield, yield_reason, and absolute_gain."""
    best_speedup = current_summary.get("best_speedup", 0.0)
    if not isinstance(best_speedup, (int, float)):
        best_speedup = 0.0

    baseline_speedup = 0.0
    if baseline_entry:
        bs = baseline_entry.get("best_speedup", 0.0)
        if isinstance(bs, (int, float)):
            baseline_speedup = bs

    target_speedup = current_summary.get("target_speedup", 5.0)
    if not isinstance(target_speedup, (int, float)):
        target_speedup = 5.0

    perf_data = current_summary.get("perf_data") or {}
    total = perf_data.get("total_cases", 0)
    passed = perf_data.get("passed_cases", 0)
    verify_passed = (
        isinstance(total, (int, float))
        and isinstance(passed, (int, float))
        and int(total) > 0
        and int(passed) == int(total)
    )

    improvement_made = bool(current_summary.get("optimized", False))

    yield_label, yield_reason, absolute_gain = strategy.classify_direction_yield(
        best_speedup=best_speedup,
        baseline_speedup=baseline_speedup,
        target_speedup=target_speedup,
        verify_passed=verify_passed,
        improvement_made=improvement_made,
        plateau_detected=plateau_detected,
        timeout=timeout,
        idle=idle,
        context=context,
        global_baseline_speedup=global_baseline_speedup,
    )

    absolute_gain_global = (
        best_speedup - global_baseline_speedup
        if global_baseline_speedup > 0
        else 0.0
    )

    return {
        "direction_yield": yield_label,
        "yield_reason": yield_reason,
        "absolute_gain_vs_baseline": round(absolute_gain, 6),
        "absolute_gain_vs_global_baseline": round(absolute_gain_global, 6),
    }


def _select_baseline_path(
    op_dir: Path,
    op_name: str,
    baseline_dir_name: str | None,
    baseline_dir_names: list[str],
) -> tuple[Path | None, dict[str, Any] | None, float]:
    """Select the baseline code path and matching round_index entry."""
    round_index = _load_round_index(op_dir)

    # Prefer explicit single baseline.
    if baseline_dir_name:
        candidate = op_dir / baseline_dir_name / f"{op_name}_generated.py"
        if candidate.is_file():
            entry = _entry_for_baseline(round_index, baseline_dir_name)
            speedup = float(entry.get("best_speedup", 0.0)) if entry else 0.0
            return candidate, entry, speedup
        return None, None, 0.0

    # If multiple baselines are given, pick the first valid one.
    for name in baseline_dir_names:
        if not name:
            continue
        candidate = op_dir / name / f"{op_name}_generated.py"
        if candidate.is_file():
            entry = _entry_for_baseline(round_index, name)
            speedup = float(entry.get("best_speedup", 0.0)) if entry else 0.0
            return candidate, entry, speedup

    return None, None, 0.0


def _entry_for_baseline(round_index: list[dict[str, Any]], round_dir: str) -> dict[str, Any] | None:
    for entry in round_index:
        if entry.get("round_dir") == round_dir:
            return entry
    return None


def transition_to_next_round(
    op_dir: Path | str,
    current_round: int,
    next_round_strategy: str,
    next_analysis_policy: str,
    next_hypothesis: str,
    next_direction: str,
    next_evidence_sources: list[str],
    current_success_criteria: str = "",
    current_key_learnings: str = "",
    baseline_dir: str | None = None,
    baseline_dirs: list[str] | None = None,
    combination_rule: str | None = None,
    switch_marker: SwitchMarker | None = None,
    strategy_name: str = "naive",
    strategy_params: dict[str, Any] | None = None,
    global_baseline_dir: str | None = None,
    global_baseline_speedup: float = 1.0,
    algorithm: str = "naive",
    algorithm_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically execute the round transition."""
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)

    result: dict[str, Any] = {
        "status": "fail",
        "next_round": current_round + 1,
        "next_round_dir": f"opt-round-{current_round + 1}",
        "baseline_speedup": 0.0,
        "global_baseline_speedup": global_baseline_speedup,
        "global_baseline_dir": global_baseline_dir,
        "issues": [],
        "warnings": [],
    }

    # Resolve global baseline speedup from baseline_info.json if not provided.
    if global_baseline_dir:
        baseline_info_path = op_dir / global_baseline_dir / "baseline_info.json"
        if baseline_info_path.is_file():
            try:
                baseline_info = load_json(baseline_info_path)
                gbs = baseline_info.get("baseline_speedup", 0.0)
                if isinstance(gbs, (int, float)) and gbs > 0:
                    global_baseline_speedup = float(gbs)
            except (ValueError, json.JSONDecodeError):
                pass
    else:
        baseline_info_path = op_dir / "global_baseline" / "baseline_info.json"
        if baseline_info_path.is_file():
            try:
                baseline_info = load_json(baseline_info_path)
                gbs = baseline_info.get("baseline_speedup", 0.0)
                if isinstance(gbs, (int, float)) and gbs > 0:
                    global_baseline_speedup = float(gbs)
                    global_baseline_dir = "global_baseline"
            except (ValueError, json.JSONDecodeError):
                pass

    result["global_baseline_speedup"] = global_baseline_speedup
    result["global_baseline_dir"] = global_baseline_dir

    issues, current_summary = _validate_current_round(op_dir / f"opt-round-{current_round}")
    if issues:
        result["issues"] = issues
        return result

    # Read existing round_index.
    existing_rounds = _load_round_index(op_dir)

    # Compute next round index (avoid colliding with the current round if the
    # round index is empty or stale).
    max_existing = max((r.get("round_index", 0) for r in existing_rounds), default=0)
    next_round_index = max(current_round, max_existing) + 1

    # Determine directory name: always a normal opt-round-{N+1} directory.
    source_round_dir = f"opt-round-{current_round}"
    next_round_dir_name = f"opt-round-{next_round_index}"
    result["next_round_dir"] = next_round_dir_name

    next_round_dir = op_dir / next_round_dir_name
    next_round_dir.mkdir(parents=True, exist_ok=True)

    # Load the strategy for the selected algorithm to classify direction yield.
    # Explicit strategy_params are honored when provided; otherwise the strategy
    # parameters are read from config["algorithms"][algorithm].params.
    config = read_config(op_dir.parent.parent)
    strategy = load_strategy_for_algorithm(config, algorithm, params=strategy_params)

    # Select baseline.
    op_name = find_op_name(op_dir)
    baseline_path: Path | None = None
    baseline_speedup = 0.0
    baseline_entry: dict[str, Any] | None = None

    if baseline_dir or baseline_dirs:
        baseline_path, baseline_entry, baseline_speedup = _select_baseline_path(
            op_dir,
            op_name or "",
            baseline_dir,
            baseline_dirs or [],
        )
        if baseline_path is None:
            result["warnings"].append(
                "specified baseline not found; next round starts without baseline code"
            )
        else:
            result["warnings"].append(
                f"baseline copied from {baseline_entry.get('round_dir') if baseline_entry else 'unknown'} "
                f"(speedup={baseline_speedup:.4f})"
            )
    else:
        result["warnings"].append("no baseline specified, next round starts without baseline code")

    result["baseline_speedup"] = baseline_speedup

    if baseline_path and baseline_path.is_file() and op_name:
        target_path = next_round_dir / f"{op_name}_generated.py"
        shutil.copy2(baseline_path, target_path)

        # Copy algorithm-specific state files if the baseline is a previous round.
        if baseline_entry and baseline_entry.get("round_dir"):
            source_round_dir_path = op_dir / baseline_entry["round_dir"]
            copied_state = _copy_algorithm_state_files(
                source_round_dir_path,
                next_round_dir,
                op_name,
                strategy,
            )
            if copied_state:
                result["warnings"].append(
                    f"copied algorithm state files for {algorithm}: {copied_state}"
                )
    elif op_name:
        # Ensure a generated code file exists even if starting from scratch.
        target_path = next_round_dir / f"{op_name}_generated.py"
        if not target_path.is_file():
            target_path.write_text("# placeholder\n", encoding="utf-8")

    # Compute direction metadata for the *completed* current round.
    current_round_dir = op_dir / source_round_dir
    plateau_detected = (
        switch_marker is not None
        and switch_marker.reason == "plateau_stagnation"
    )
    direction_metadata = _compute_direction_metadata(
        current_summary,
        baseline_entry,
        strategy=strategy,
        plateau_detected=plateau_detected,
        timeout=current_summary.get("status") == "timeout",
        idle=(current_summary.get("round_type") == "idle"),
        context={"op_dir": str(op_dir), "round_dir": str(current_round_dir)},
        global_baseline_speedup=global_baseline_speedup,
    )

    # Write next summary.json.
    next_summary = {
        "round_index": next_round_index,
        "algorithm": algorithm,
        "baseline_dir": baseline_entry.get("round_dir") if baseline_entry else None,
        "global_baseline_dir": global_baseline_dir,
        "global_baseline_speedup": global_baseline_speedup,
        "round_strategy": next_round_strategy,
        "analysis_policy": next_analysis_policy,
        "hypothesis": next_hypothesis,
        "direction": next_direction,
        "evidence_sources": next_evidence_sources,
        "status": "active",
        "best_speedup": 0.0,
        "success": False,
        "gen_iterations": 0,
        "opt_iterations": 0,
        "optimized": False,
        "target_reached": False,
        "target_speedup": current_summary.get("target_speedup", 5.0),
        "direction_yield": None,
        "yield_reason": None,
        "absolute_gain_vs_baseline": None,
        "absolute_gain_vs_global_baseline": None,
    }

    # Persist algorithm-specific metadata so the next round's worker can read it.
    if algorithm_metadata:
        for key in ("mcts_parent_id", "mcts_selection_path", "mcts_pending_expansion"):
            if key in algorithm_metadata:
                next_summary[key] = algorithm_metadata[key]

    (next_round_dir / "summary.json").write_text(
        json.dumps(next_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Update state file.
    state_dir_loc = _find_state_dir(op_dir)
    if state_dir_loc is None:
        result["warnings"].append(".triton-agent/ not found, state file will not be updated")
    op_name = find_op_name(op_dir) or (op_dir.name if op_dir.name else None)
    state_path = _state_path_for_op_dir(op_dir, state_dir_loc, op_name)
    if state_path and state_path.is_file():
        try:
            state = load_json(state_path)
            if str(current_round) in state.get("rounds", {}):
                state["rounds"][str(current_round)]["status"] = "completed"
            rounds = state.setdefault("rounds", {})
            rounds[str(next_round_index)] = {
                "status": "active",
                "round_dir": next_round_dir_name,
            }
            state["current_round"] = next_round_index
            state["algorithm"] = algorithm
            state["algorithm_reference"] = f".claude/references/algorithms/{algorithm}/round.md"
            state["status"] = "active"
            state["phase"] = "round_active"
            state["work_dir"] = str(next_round_dir)
            state["last_phase"] = 0
            # Maintain direction_history for fast recovery.
            state["direction_history"] = build_direction_history(existing_rounds)
            state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except (ValueError, json.JSONDecodeError) as exc:
            result["warnings"].append(f"state file update failed: {exc}")

    # Update round_index.json: finalize current round entry and add next round entry.
    current_entry = _entry_for_round_index(existing_rounds, current_round)
    if current_entry is None:
        current_entry = {
            "round_index": current_round,
            "round_dir": source_round_dir,
            "direction": current_summary.get("direction", ""),
            "hypothesis": current_summary.get("hypothesis", ""),
            "evidence_sources": current_summary.get("evidence_sources", []),
            "analysis_policy": current_summary.get("analysis_policy", "pattern_entry"),
            "effective_metric_source": current_summary.get("effective_metric_source", "kernel"),
            "status": "success" if current_summary.get("success", False) else "failed",
            "best_speedup": current_summary.get("best_speedup", 0.0),
            "round_strategy": current_summary.get("round_strategy", "exploration"),
            "consecutive_low_gain_count": current_summary.get("consecutive_low_gain_count", 0),
            "round_type": current_summary.get("round_type", "active"),
            "target_speedup": current_summary.get("target_speedup", 5.0),
            "global_baseline_dir": current_summary.get("global_baseline_dir") or global_baseline_dir,
            "global_baseline_speedup": current_summary.get("global_baseline_speedup") or global_baseline_speedup,
        }
        existing_rounds.append(current_entry)

    # Merge computed direction metadata into current entry.
    current_entry.update(direction_metadata)
    current_entry["status"] = "success" if current_summary.get("success", False) else "failed"

    # Ensure the current entry records the actual baseline it started from.
    if "baseline_dir" not in current_entry:
        current_entry["baseline_dir"] = current_summary.get("baseline_dir")
    if "global_baseline_dir" not in current_entry or not current_entry["global_baseline_dir"]:
        current_entry["global_baseline_dir"] = global_baseline_dir
    if "global_baseline_speedup" not in current_entry or not current_entry["global_baseline_speedup"]:
        current_entry["global_baseline_speedup"] = global_baseline_speedup
    # Drop legacy fields in favor of baseline_dir.
    current_entry.pop("parent_round_dir", None)
    current_entry.pop("best_baseline_for_this_direction", None)

    # Build next round entry, embedding switch marker if present.
    next_entry: dict[str, Any] = {
        "round_index": next_round_index,
        "round_dir": next_round_dir_name,
        "algorithm": algorithm,
        "direction": next_direction,
        "hypothesis": next_hypothesis,
        "evidence_sources": next_evidence_sources,
        "analysis_policy": next_analysis_policy,
        "effective_metric_source": "kernel",
        "status": "active",
        "best_speedup": 0.0,
        "round_strategy": next_round_strategy,
        "consecutive_low_gain_count": 0,
        "round_type": "active",
        "target_speedup": current_summary.get("target_speedup", 5.0),
        "direction_yield": None,
        "yield_reason": None,
        "absolute_gain_vs_baseline": None,
        "absolute_gain_vs_global_baseline": None,
        "baseline_dir": baseline_entry.get("round_dir") if baseline_entry else None,
        "global_baseline_dir": global_baseline_dir,
        "global_baseline_speedup": global_baseline_speedup,
    }
    if switch_marker is not None:
        next_entry["switch_reason"] = switch_marker.reason
        next_entry["previous_direction"] = switch_marker.previous_direction
        next_entry["new_direction"] = switch_marker.new_direction

    existing_rounds.append(next_entry)

    round_index_path = op_dir / "round_index.json"
    round_index_path.write_text(
        json.dumps(existing_rounds, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write opt-note.md.
    opt_note_path = op_dir / "opt-note.md"
    opt_entry = (
        f"\n## Round {current_round}\n\n"
        f"- round_strategy: {current_summary.get('round_strategy', 'N/A')}\n"
        f"- hypothesis: {current_summary.get('hypothesis', 'N/A')}\n"
        f"- direction: {current_summary.get('direction', 'N/A')}\n"
        f"- success_criteria: {current_success_criteria}\n"
        f"- evidence_sources: {json.dumps(current_summary.get('evidence_sources', []), ensure_ascii=False)}\n"
        f"- result: {'success' if current_summary.get('success', False) else 'failed'}\n"
        f"- best_speedup: {current_summary.get('best_speedup', 0)}\n"
        f"- direction_yield: {direction_metadata.get('direction_yield')}\n"
        f"- yield_reason: {direction_metadata.get('yield_reason')}\n"
        f"- absolute_gain_vs_baseline: {direction_metadata.get('absolute_gain_vs_baseline')}\n"
        f"- absolute_gain_vs_global_baseline: {direction_metadata.get('absolute_gain_vs_global_baseline')}\n"
    )
    if switch_marker is not None:
        opt_entry += (
            f"- switch_reason: {switch_marker.reason}\n"
            f"- previous_direction: {switch_marker.previous_direction}\n"
            f"- new_direction: {switch_marker.new_direction}\n"
        )
    opt_entry += f"- key_learnings: {current_key_learnings}\n"

    with open(opt_note_path, "a", encoding="utf-8") as f:
        f.write(opt_entry)

    # Write transition lock.
    lock = {
        "action": "continue",
        "transition_at": datetime.now(timezone.utc).isoformat(),
        "from_round": current_round,
        "to_round": next_round_index,
        "to_round_dir": next_round_dir_name,
        "state_after": {
            "current_round": next_round_index,
            "last_phase": 0,
            "status": "active",
            "phase": "round_active",
        },
    }
    lock_path = op_dir / ".transition_lock.json"
    try:
        lock_path.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        result["warnings"].append(f"failed to write transition lock: {exc}")

    result["status"] = "success"
    result["next_round"] = next_round_index
    result["issues"] = issues
    result["warnings"] = result["warnings"]
    return result


def _resolve_algorithm(
    cli_algorithm: str | None,
    strategy_name: str,
    op_dir: Path,
) -> str:
    """Resolve the algorithm when --algorithm is not provided.

    Priority:
      1. Explicit CLI --algorithm.
      2. The single enabled algorithm in config.json.
      3. The algorithm saved in the current workspace state file, if enabled.
      4. The explicit --strategy-name (when not auto).
    """
    if cli_algorithm is not None:
        return cli_algorithm

    config = read_config(op_dir.parent.parent)
    discovered = set(discover_strategies().keys())
    enabled = [a for a in get_enabled_algorithms(config) if a in discovered]

    if len(enabled) == 1:
        return enabled[0]

    state_dir = _find_state_dir(op_dir)
    if state_dir is not None:
        state_path = state_dir / f"state-{op_dir.name}.json"
        if state_path.is_file():
            try:
                saved_algo = load_json(state_path).get("algorithm")
                if isinstance(saved_algo, str) and saved_algo in enabled:
                    return saved_algo
            except ValueError:
                pass

    if strategy_name != "auto" and strategy_name in discovered:
        return strategy_name

    if not enabled:
        raise ValueError(
            "No algorithms are enabled in config.json; pass --algorithm explicitly."
        )
    raise ValueError(
        f"Multiple algorithms are enabled in config.json: {enabled}. "
        "Pass --algorithm explicitly."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Transition to the next optimization round")
    parser.add_argument("--op-dir", required=True, help="Operator directory path")
    parser.add_argument("--current-round", type=int, required=True, help="Current round number")
    parser.add_argument("--next-round-strategy", required=True,
                        choices=["exploration", "structural_change", "focused_tuning", "stabilization", "plateau_review"],
                        help="Next round strategy")
    parser.add_argument("--next-analysis-policy", required=True,
                        choices=["pattern_entry", "bottleneck_analysis", "ir_supported"],
                        help="Next round analysis policy")
    parser.add_argument("--next-hypothesis", required=True, help="Next round hypothesis")
    parser.add_argument("--next-direction", required=True, help="Next round direction")
    parser.add_argument("--next-evidence-sources", required=True, help="JSON list of evidence sources")
    parser.add_argument("--current-success-criteria", default="", help="Current round success criteria")
    parser.add_argument("--current-key-learnings", default="", help="Current round key learnings")
    parser.add_argument("--baseline-dir", default=None,
                        help="Single baseline round directory to copy code from")
    parser.add_argument("--baseline-dirs", default=None,
                        help="JSON list of baseline round directories (multiple baseline mode)")
    parser.add_argument("--combination-rule", default=None,
                        help="How to combine multiple baselines (e.g. crossover)")
    parser.add_argument("--global-baseline-dir", default=None,
                        help="Global baseline directory for speedup reference")
    parser.add_argument("--global-baseline-speedup", type=float, default=1.0,
                        help="Global baseline speedup (default 1.0)")
    parser.add_argument("--switch-marker", default=None,
                        help='JSON object with reason/previous_direction/new_direction for audit')
    parser.add_argument("--strategy-name", default="auto",
                        help="Direction/baseline selection strategy name. "
                             "Use 'auto' to load the strategy configured in config.json "
                             "(defaults to naive if no config is found).")
    parser.add_argument("--strategy-params", default="{}",
                        help="JSON object of strategy-specific parameters. "
                             "Ignored when --strategy-name=auto.")
    parser.add_argument("--algorithm", default=None,
                        help="Optimization algorithm for the next round. "
                             "Required unless --strategy-name=auto and config.json enables "
                             "exactly one algorithm.")
    parser.add_argument("--algorithm-metadata", default="{}",
                        help="JSON object with algorithm-specific metadata to persist in next summary.json")
    args = parser.parse_args()

    op_dir = Path(args.op_dir).expanduser().resolve()
    if not op_dir.is_dir():
        print(json.dumps({
            "status": "fail",
            "next_round": args.current_round + 1,
            "next_round_dir": f"opt-round-{args.current_round + 1}",
            "baseline_speedup": 0.0,
            "issues": [f"operator directory not found: {op_dir}"],
            "warnings": [],
        }))
        return 1

    try:
        algorithm = _resolve_algorithm(args.algorithm, args.strategy_name, op_dir)
    except ValueError as exc:
        print(json.dumps({
            "status": "fail",
            "next_round": args.current_round + 1,
            "next_round_dir": f"opt-round-{args.current_round + 1}",
            "baseline_speedup": 0.0,
            "issues": [str(exc)],
            "warnings": [],
        }))
        return 1

    # Strategy parameters:
    # Explicit --strategy-params takes precedence. When --strategy-name=auto
    # (the normal Orchestrator call path), we still honor an explicit params
    # JSON and only fall back to config.json if it is empty/missing/invalid.
    try:
        strategy_params = json.loads(args.strategy_params)
        if not isinstance(strategy_params, dict):
            strategy_params = {}
    except (json.JSONDecodeError, TypeError):
        strategy_params = {}

    if args.strategy_name == "auto":
        strategy_name = algorithm
        if not strategy_params:
            config = read_config(op_dir.parent.parent)
            strategy_params = get_algorithm_params(config, algorithm)
    else:
        strategy_name = args.strategy_name

    try:
        evidence_sources = json.loads(args.next_evidence_sources)
        if not isinstance(evidence_sources, list):
            evidence_sources = [args.next_evidence_sources]
    except (json.JSONDecodeError, TypeError):
        evidence_sources = [args.next_evidence_sources]

    baseline_dirs: list[str] = []
    if args.baseline_dirs:
        try:
            parsed = json.loads(args.baseline_dirs)
            if isinstance(parsed, list):
                baseline_dirs = [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            baseline_dirs = []

    switch_marker: SwitchMarker | None = None
    if args.switch_marker:
        try:
            data = json.loads(args.switch_marker)
            switch_marker = SwitchMarker(
                reason=data.get("reason", ""),
                previous_direction=data.get("previous_direction"),
                new_direction=data.get("new_direction"),
            )
        except (json.JSONDecodeError, TypeError):
            pass

    try:
        algorithm_metadata = json.loads(args.algorithm_metadata)
        if not isinstance(algorithm_metadata, dict):
            algorithm_metadata = {}
    except (json.JSONDecodeError, TypeError):
        algorithm_metadata = {}

    result = transition_to_next_round(
        op_dir,
        current_round=args.current_round,
        next_round_strategy=args.next_round_strategy,
        next_analysis_policy=args.next_analysis_policy,
        next_hypothesis=args.next_hypothesis,
        next_direction=args.next_direction,
        next_evidence_sources=evidence_sources,
        current_success_criteria=args.current_success_criteria,
        current_key_learnings=args.current_key_learnings,
        baseline_dir=args.baseline_dir,
        baseline_dirs=baseline_dirs,
        combination_rule=args.combination_rule,
        switch_marker=switch_marker,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        global_baseline_dir=args.global_baseline_dir,
        global_baseline_speedup=args.global_baseline_speedup,
        algorithm=algorithm,
        algorithm_metadata=algorithm_metadata,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
