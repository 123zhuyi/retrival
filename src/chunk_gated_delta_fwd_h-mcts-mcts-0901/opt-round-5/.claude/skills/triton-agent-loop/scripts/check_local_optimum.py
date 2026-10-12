#!/usr/bin/env python3
"""
Local Optimum Detection and Direction Coverage Check

Detects whether the optimization has plateaued (local optimum) by checking
recent rounds for diminishing returns, and tracks direction coverage
for the Agent Loop's stop/switch decision system.

Usage:
    python3 check_local_optimum.py \
        --op-dir triton_ascend_output/{op_name}-{algorithm}-{run_tag} \
        --window 3 --max-gain 0.02

Output (JSON):
    {
        "in_local_optimum": false,
        "window_size": 3,
        "max_gain": 0.02,
        "gains": [...],
        "warnings": [],
        "direction_coverage": { ... }
    }
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from direction_selector import get_low_yield_directions


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


def _read_round_summaries(op_dir: Path) -> list[dict[str, Any]]:
    """Read summary.json from each opt-round-* directory."""
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
            "status": "success" if data.get("success", False) else "unknown",
            "direction_yield": data.get("direction_yield"),
            "yield_reason": data.get("yield_reason"),
        })
    return rounds


def check_local_optimum(
    op_dir: Path | str,
    window: int = 3,
    max_gain: float = 0.02,
) -> dict[str, Any]:
    """Check if recent rounds show diminishing returns (local optimum).

    Reads round data from opt-round-*/summary.json and calculates
    the gain between consecutive rounds. If all gains within the
    sliding window are below max_gain, the optimizer is in a local optimum.

    Args:
        op_dir: Operator directory containing opt-round-* directories.
        window: Number of recent rounds to check (default: 3).
        max_gain: Maximum gain threshold (default: 0.02 = 2%).

    Returns:
        dict with in_local_optimum, gains, warnings.
    """
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)

    result: dict[str, Any] = {
        "in_local_optimum": False,
        "window_size": window,
        "max_gain": max_gain,
        "gains": [],
        "warnings": [],
        "direction_coverage": {},
    }

    if not op_dir.is_dir():
        result["warnings"].append(f"op_dir not found: {op_dir}")
        return result

    rounds = _read_round_summaries(op_dir)

    if len(rounds) < 2:
        result["warnings"].append(f"insufficient rounds ({len(rounds)}) for local optimum detection")
        result["direction_coverage"] = check_direction_coverage(op_dir, rounds)
        return result

    # Calculate gains between consecutive rounds (only for rounds with valid speedups)
    gains = []
    gain_details = []
    for i in range(1, len(rounds)):
        prev = rounds[i - 1]["best_speedup"]
        curr = rounds[i]["best_speedup"]
        if isinstance(prev, (int, float)) and isinstance(curr, (int, float)) and prev > 0 and curr > 0:
            gain = (curr - prev) / prev
            gains.append(gain)
            gain_details.append({
                "from": rounds[i - 1]["round_dir"],
                "to": rounds[i]["round_dir"],
                "from_speedup": prev,
                "to_speedup": curr,
                "gain": gain,
            })
        else:
            gains.append(None)
            gain_details.append({
                "from": rounds[i - 1]["round_dir"],
                "to": rounds[i]["round_dir"],
                "from_speedup": prev,
                "to_speedup": curr,
                "gain": None,
            })

    result["gains"] = gain_details

    # Check the last `window` gains
    valid_gains = [g for g in gains[-window:] if g is not None]
    if len(valid_gains) < window:
        result["warnings"].append(
            f"insufficient valid gains in window: need {window}, got {len(valid_gains)}"
        )
        if len(valid_gains) >= 2:
            result["in_local_optimum"] = all(g < max_gain for g in valid_gains)
    else:
        result["in_local_optimum"] = all(g < max_gain for g in valid_gains)

    # Direction coverage
    result["direction_coverage"] = check_direction_coverage(op_dir, rounds)

    # Per-direction consecutive low-gain count.
    result["per_direction_low_gain"] = _compute_per_direction_low_gain(rounds, max_gain)

    return result


def _compute_per_direction_low_gain(
    rounds: list[dict[str, Any]],
    max_gain: float,
) -> dict[str, int]:
    """Count consecutive low-gain rounds per direction.

    A round counts as low-gain for its direction if the gain from the previous
    round is below max_gain.
    """
    counts: dict[str, int] = {}
    current_streak: dict[str, int] = {}
    for i, r in enumerate(rounds):
        direction = r.get("direction", "")
        if not direction:
            continue
        if i == 0:
            current_streak[direction] = 0
            continue
        prev_speedup = rounds[i - 1].get("best_speedup", 0)
        curr_speedup = r.get("best_speedup", 0)
        if (
            isinstance(prev_speedup, (int, float))
            and isinstance(curr_speedup, (int, float))
            and prev_speedup > 0
            and curr_speedup > 0
        ):
            gain = (curr_speedup - prev_speedup) / prev_speedup
            if gain < max_gain:
                current_streak[direction] = current_streak.get(direction, 0) + 1
            else:
                current_streak[direction] = 0
        else:
            current_streak[direction] = current_streak.get(direction, 0) + 1
        counts[direction] = max(counts.get(direction, 0), current_streak.get(direction, 0))
    return counts


def check_direction_coverage(
    op_dir: Path | str,
    rounds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize historically attempted optimization directions.

    With pattern_index.md removed, there is no static catalog. The result only
    describes directions that have actually been tried, plus their latest yield.
    The old ``total_patterns`` / ``coverage_ratio`` / ``untried_patterns``
    fields are omitted because a fixed catalog no longer exists; reporting them
    as zero would be misleading.

    Returns:
        dict with attempted_patterns, low_yield_patterns, attempt_details,
        and direction_yields.
    """
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)

    if rounds is None:
        rounds = _read_round_summaries(op_dir)

    # Collect attempted directions and their latest yield.
    attempted: set[str] = set()
    direction_yields: dict[str, str] = {}
    direction_details = []
    for r in rounds:
        direction = r.get("direction", "")
        if direction and direction != "no_change":
            attempted.add(direction)
            direction_yields[direction] = r.get("direction_yield") or direction_yields.get(direction, "")
            direction_details.append({
                "direction": direction,
                "round_dir": r["round_dir"],
                "status": r.get("status", "unknown"),
                "yield": r.get("direction_yield"),
            })

    low_yield = get_low_yield_directions(rounds)

    return {
        "attempted_patterns": sorted(list(attempted)),
        "low_yield_patterns": sorted(list(low_yield)),
        "attempt_details": direction_details,
        "direction_yields": direction_yields,
    }


def check_direction_exhaustion(
    op_dir: Path | str,
    rounds: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check whether all attempted optimization directions are exhausted.

    A direction is considered exhausted only when it has been tried at all
    three analysis_policy levels (pattern_entry → bottleneck_analysis →
    ir_supported) and every attempt was low-yield or failed.

    With pattern_index.md removed, known directions are only those that have
    actually been attempted in this optimization session. There is therefore no
    separate "untried" set; a direction is either active or exhausted.

    Returns:
        dict with direction_exhausted, exhausted_directions, active_directions,
        known_directions, and per-direction policy coverage details.
    """
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)

    if rounds is None:
        rounds = _read_round_summaries(op_dir)

    policy_levels = ["pattern_entry", "bottleneck_analysis", "ir_supported"]

    # Known directions are only those already attempted in this session.
    known_patterns: set[str] = set()

    # Collect attempts per direction, grouped by analysis_policy.
    attempts_by_direction: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rounds:
        direction = r.get("direction", "")
        if not direction or direction == "no_change":
            continue
        policy = r.get("analysis_policy", "pattern_entry") or "pattern_entry"
        attempts_by_direction[direction][policy].append(r)
        known_patterns.add(direction)

    exhausted: set[str] = set()
    active: set[str] = set()
    for direction in known_patterns:
        policy_attempts = attempts_by_direction.get(direction, {})
        if not all(level in policy_attempts for level in policy_levels):
            active.add(direction)
            continue
        all_low = all(
            r.get("direction_yield") in ("low", "failed")
            for level in policy_levels
            for r in policy_attempts[level]
        )
        if all_low:
            exhausted.add(direction)
        else:
            active.add(direction)

    direction_exhausted = bool(known_patterns) and all(d in exhausted for d in known_patterns)

    return {
        "direction_exhausted": direction_exhausted,
        "exhausted_directions": sorted(exhausted),
        "active_directions": sorted(active),
        "known_directions": sorted(known_patterns),
        "per_direction_policy": {
            d: {level: len(attempts_by_direction[d].get(level, [])) for level in policy_levels}
            for d in sorted(known_patterns)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check local optimum and direction coverage"
    )
    parser.add_argument("--op-dir", required=True, help="Operator directory path")
    parser.add_argument("--window", type=int, default=3, help="Window size for gain check (default: 3)")
    parser.add_argument("--max-gain", type=float, default=0.02, help="Max gain threshold (default: 0.02)")
    args = parser.parse_args()

    op_dir = Path(args.op_dir).expanduser().resolve()
    if not op_dir.is_dir():
        print(json.dumps({
            "in_local_optimum": False,
            "warnings": [f"op_dir not found: {op_dir}"],
        }))
        return 1

    result = check_local_optimum(op_dir, args.window, args.max_gain)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
