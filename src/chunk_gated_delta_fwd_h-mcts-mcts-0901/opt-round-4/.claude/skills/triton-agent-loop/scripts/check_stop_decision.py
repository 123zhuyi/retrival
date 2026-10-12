#!/usr/bin/env python3
"""Stop Decision Validation Script

Agent MUST run this script BEFORE outputting any "stop" or "final report" decision.
The script objectively verifies whether stop conditions (C1/C2) are met and
whether the active strategy recommends stopping, preventing subjective early
termination of the optimization loop.

Usage:
    python3 check_stop_decision.py \
        --op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag} \
        --current-round {N} \
        [--best-speedup X] \
        [--target-speedup Y] \
        [--max-rounds M] \
        [--algorithm-recommends-stop]

Output (JSON):
    {
        "can_stop": false,
        "reason": "no stop condition met",
        "conditions_met": [],
        "checks": { ... }
    }

Rules:
    - Agent MUST NOT stop when can_stop is false
    - Agent MUST NOT output a final summary report when can_stop is false
    - Agent MUST NOT ask the user for confirmation to continue when can_stop is false
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


def _read_rounds_from_summaries(op_dir: Path) -> list[dict[str, Any]]:
    """Read summary.json from each opt-round-* directory.

    Sorts directories by numeric round index and parses directories
    like opt-round-9-switch correctly (kept for backwards compatibility).
    """
    rounds = []
    for rd in sorted(
        op_dir.glob("opt-round-*"),
        key=lambda p: int(p.name.replace("opt-round-", "").split("-")[0])
        if p.name.replace("opt-round-", "").split("-")[0].isdigit()
        else 0,
    ):
        summary_path = rd / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            data = load_json(summary_path)
        except ValueError:
            continue

        speedup = data.get("best_speedup", 0)
        if not isinstance(speedup, (int, float)):
            speedup = 0

        rounds.append({
            "round_index": data.get("round_index", 0),
            "round_dir": rd.name,
            "best_speedup": speedup,
            "round_strategy": data.get("round_strategy", ""),
            "direction": data.get("direction", ""),
            "analysis_policy": data.get("analysis_policy", "pattern_entry"),
            "status": "success" if data.get("success", False) else ("active" if data.get("failure_phase") else "unknown"),
            "improvement_made": data.get("optimized", False),
            "phase4_entered": data.get("phase4_entered", False),
        })
    return rounds


def check_stop_decision(
    op_dir: Path | str,
    current_round: int = 0,
    best_speedup: float = 0.0,
    target_speedup: float = 1.0,
    max_rounds: int = 5,
    algorithm_recommends_stop: bool = False,
) -> dict[str, Any]:
    """Objectively evaluate C1/C2 stop conditions and strategy stop recommendation.

    Reads round data DIRECTLY from opt-round-*/summary.json files.
    Does NOT trust round_index.json (which the model can manipulate).

    Returns a structured decision that the Agent MUST obey.
    """
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)
    result: dict[str, Any] = {
        "can_stop": False,
        "reason": "",
        "conditions_met": [],
        "checks": {},
    }

    # ── Read round data from summary.json (authoritative source) ──
    all_rounds = _read_rounds_from_summaries(op_dir)

    # Collect round stats from actual data
    non_idle_entries = [r for r in all_rounds if r.get("best_speedup", 0) > 0 or r.get("phase4_entered", False)]

    # Compute best_speedup from actual data (overrides CLI arg)
    actual_best_speedup = max((r.get("best_speedup", 0) for r in non_idle_entries), default=0)
    if best_speedup > 0 and actual_best_speedup > 0 and abs(best_speedup - actual_best_speedup) > 0.001:
        result.setdefault("warnings", []).append(
            f"CLI best_speedup({best_speedup:.4f}) differs from actual max from summary.json({actual_best_speedup:.4f}). Using actual value."
        )
    effective_best_speedup = max(best_speedup, actual_best_speedup)

    # ═══════════════════════════════════════════════════════
    # C1: Target Reached
    # ═══════════════════════════════════════════════════════
    c1_met = False
    c1_detail = ""
    if isinstance(effective_best_speedup, (int, float)) and isinstance(target_speedup, (int, float)):
        if math.isfinite(effective_best_speedup) and target_speedup > 0:
            c1_met = effective_best_speedup >= target_speedup
            c1_detail = (
                f"best_speedup({effective_best_speedup:.4f}) "
                f"{'>=' if c1_met else '<'}"
                f" target_speedup({target_speedup})"
            )
        else:
            c1_detail = f"invalid values: best_speedup={effective_best_speedup}, target_speedup={target_speedup}"
    else:
        c1_detail = f"non-numeric values: best_speedup={effective_best_speedup}, target_speedup={target_speedup}"

    # ═══════════════════════════════════════════════════════
    # C2: Max Rounds Reached
    # ═══════════════════════════════════════════════════════
    non_idle_count = len(non_idle_entries)
    c2_met = non_idle_count >= max_rounds
    c2_detail = f"non-idle rounds({non_idle_count}) {'>=' if c2_met else '<'} max_rounds({max_rounds})"

    # ═══════════════════════════════════════════════════════
    # Strategy-level stop recommendation
    # ═══════════════════════════════════════════════════════
    strategy_stop_met = algorithm_recommends_stop
    strategy_detail = (
        f"algorithm_recommends_stop={algorithm_recommends_stop}"
    )

    # ── Assemble result ──
    conditions_met: list[str] = []
    if c1_met:
        conditions_met.append("C1")
    if c2_met:
        conditions_met.append("C2")
    if strategy_stop_met:
        conditions_met.append("strategy_stop")

    result["conditions_met"] = conditions_met
    result["checks"] = {
        "C1": {"met": c1_met, "detail": c1_detail},
        "C2": {"met": c2_met, "detail": c2_detail},
        "strategy_stop": {"met": strategy_stop_met, "detail": strategy_detail},
    }

    if c1_met:
        result["can_stop"] = True
        result["reason"] = "C1: target_reached"
    elif c2_met:
        result["can_stop"] = True
        result["reason"] = "C2: max_rounds_reached"
    elif strategy_stop_met:
        result["can_stop"] = True
        result["reason"] = "strategy_recommends_stop"
    else:
        result["can_stop"] = False
        result["reason"] = "no stop condition met (C1/C2 not met)"

    # ── Write transition lock on stop decision (releases Phase 7 gate) ──
    if result["can_stop"]:
        lock = {
            "action": "stop",
            "transition_at": datetime.now(timezone.utc).isoformat(),
            "reason": result["reason"],
            "from_round": current_round,
            "state_after": {
                "current_round": current_round,
                "last_phase": 7,
                "status": "completed" if current_round > 0 else "active",
            },
            "conditions_met": conditions_met,
        }
        lock_path = op_dir / ".transition_lock.json"
        try:
            lock_path.write_text(
                json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8"
            )
        except OSError as exc:
            result.setdefault("warnings", []).append(f"failed to write stop lock: {exc}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate stop decision for Agent Loop (must run before any stop output)"
    )
    parser.add_argument("--op-dir", required=True, help="Operator directory path")
    parser.add_argument("--current-round", type=int, default=0, help="Current round number")
    parser.add_argument("--best-speedup", type=float, default=0.0, help="Best speedup achieved")
    parser.add_argument("--target-speedup", type=float, default=1.0, help="Target speedup")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum rounds (config.json max_rounds)")
    parser.add_argument("--algorithm-recommends-stop", action="store_true",
                        help="The active strategy recommends stopping")
    args = parser.parse_args()

    op_dir = Path(args.op_dir).expanduser().resolve()
    if not op_dir.is_dir():
        print(json.dumps({
            "can_stop": False,
            "reason": f"operator directory not found: {op_dir}",
            "conditions_met": [],
            "checks": {},
        }))
        return 1

    result = check_stop_decision(
        op_dir,
        current_round=args.current_round,
        best_speedup=args.best_speedup,
        target_speedup=args.target_speedup,
        max_rounds=args.max_rounds,
        algorithm_recommends_stop=args.algorithm_recommends_stop,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["can_stop"] else 1


if __name__ == "__main__":
    sys.exit(main())
