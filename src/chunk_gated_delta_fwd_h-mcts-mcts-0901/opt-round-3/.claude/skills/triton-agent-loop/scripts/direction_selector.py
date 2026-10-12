#!/usr/bin/env python3
"""direction_selector.py — Direction yield classification and baseline scoring.

Provides:
- Direction yield classification
- Multi-factor baseline scoring
- Next-direction recommendation (LLM-driven)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


# ---------------------------------------------------------------------------
# Direction yield classification
# ---------------------------------------------------------------------------

YieldLabel = str  # "high" | "medium" | "low" | "failed"
YieldReason = str


def classify_direction_yield(
    best_speedup: float,
    baseline_speedup: float,
    target_speedup: float,
    verify_passed: bool,
    improvement_made: bool,
    plateau_detected: bool = False,
    timeout: bool = False,
    idle: bool = False,
    context: dict[str, Any] | None = None,
    global_baseline_speedup: float = 0.0,
) -> tuple[YieldLabel, YieldReason, float]:
    """Classify a direction attempt.

    Returns:
        (direction_yield, yield_reason, absolute_gain)
    """
    absolute_gain = best_speedup - baseline_speedup if baseline_speedup > 0 else 0.0

    if timeout:
        return "failed", "timeout", absolute_gain

    if idle:
        return "failed", "idle_round", absolute_gain

    if not verify_passed:
        return "failed", "verify_failed", absolute_gain

    if best_speedup >= target_speedup:
        return "high", "target_reached", absolute_gain

    if plateau_detected:
        # Determine how many rounds of plateau based on gain magnitude.
        if absolute_gain <= 0.0:
            return "low", "no_improvement_vs_baseline", absolute_gain
        if absolute_gain < 0.05:
            return "low", "plateau_after_2_rounds", absolute_gain
        return "medium", "plateau_after_2_rounds", absolute_gain

    if not improvement_made:
        return "low", "no_improvement_vs_baseline", absolute_gain

    if absolute_gain >= 0.20:
        return "high", "improvement_made", absolute_gain
    if absolute_gain >= 0.05:
        return "medium", "improvement_made", absolute_gain
    return "low", "improvement_made", absolute_gain


def get_low_yield_directions(
    round_index: list[dict[str, Any]],
    min_attempts: int = 1,
) -> set[str]:
    """Return directions that have been marked low_yield or failed."""
    low_yield: set[str] = set()
    attempts: dict[str, int] = {}
    for entry in round_index:
        direction = entry.get("direction", "")
        if not direction:
            continue
        attempts[direction] = attempts.get(direction, 0) + 1
        yield_label = entry.get("direction_yield", "")
        if yield_label in ("low", "failed"):
            low_yield.add(direction)
    # Only mark as low-yield if there were enough attempts.
    return {d for d in low_yield if attempts.get(d, 0) >= min_attempts}


# ---------------------------------------------------------------------------
# Baseline scoring
# ---------------------------------------------------------------------------

# Compatibility matrix: baseline direction -> target direction -> score 0.0-1.0.
# Higher means the baseline already contains useful structure for the target direction.
_DIRECTION_COMPATIBILITY: dict[str, dict[str, float]] = {
    "pattern: autotune": {
        "pattern: autotune": 1.0,
        "pattern: program-multiple-rows": 0.6,
        "pattern: vectorization": 0.5,
        "pattern: pipeline": 0.3,
        "pattern: block-tiling": 0.2,
        "pattern: tl.dot": 0.1,
    },
    "pattern: program-multiple-rows": {
        "pattern: program-multiple-rows": 1.0,
        "pattern: vectorization": 0.7,
        "pattern: autotune": 0.5,
        "pattern: pipeline": 0.4,
        "pattern: block-tiling": 0.3,
        "pattern: tl.dot": 0.1,
    },
    "pattern: vectorization": {
        "pattern: vectorization": 1.0,
        "pattern: program-multiple-rows": 0.7,
        "pattern: autotune": 0.6,
        "pattern: block-tiling": 0.4,
        "pattern: pipeline": 0.3,
        "pattern: tl.dot": 0.1,
    },
    "pattern: block-tiling": {
        "pattern: block-tiling": 1.0,
        "pattern: pipeline": 0.7,
        "pattern: tl.dot": 0.6,
        "pattern: warp-specialization": 0.5,
        "pattern: vectorization": 0.4,
        "pattern: program-multiple-rows": 0.3,
        "pattern: autotune": 0.2,
    },
    "pattern: pipeline": {
        "pattern: pipeline": 1.0,
        "pattern: block-tiling": 0.7,
        "pattern: double-buffer": 0.7,
        "pattern: tl.dot": 0.6,
        "pattern: warp-specialization": 0.4,
        "pattern: program-multiple-rows": 0.3,
        "pattern: autotune": 0.2,
    },
    "pattern: double-buffer": {
        "pattern: double-buffer": 1.0,
        "pattern: pipeline": 0.8,
        "pattern: block-tiling": 0.6,
        "pattern: tl.dot": 0.5,
    },
    "pattern: tl.dot": {
        "pattern: tl.dot": 1.0,
        "pattern: block-tiling": 0.7,
        "pattern: pipeline": 0.6,
        "pattern: double-buffer": 0.5,
        "pattern: warp-specialization": 0.4,
    },
    "pattern: reduce": {
        "pattern: reduce": 1.0,
        "pattern: warp-specialization": 0.6,
        "pattern: block-tiling": 0.5,
        "pattern: program-multiple-rows": 0.4,
        "pattern: autotune": 0.3,
    },
    "pattern: warp-specialization": {
        "pattern: warp-specialization": 1.0,
        "pattern: reduce": 0.7,
        "pattern: tl.dot": 0.6,
        "pattern: pipeline": 0.5,
    },
    "pattern: kernel-fusion": {
        "pattern: kernel-fusion": 1.0,
        "pattern: elementwise": 0.7,
        "pattern: autotune": 0.4,
    },
    "pattern: elementwise": {
        "pattern: elementwise": 1.0,
        "pattern: kernel-fusion": 0.8,
        "pattern: vectorization": 0.7,
        "pattern: program-multiple-rows": 0.5,
        "pattern: autotune": 0.4,
    },
}


def compute_direction_compatibility(
    baseline_direction: str,
    target_direction: str,
) -> float:
    """Return 0.0-1.0 compatibility score."""
    if baseline_direction == target_direction:
        return 1.0
    return _DIRECTION_COMPATIBILITY.get(baseline_direction, {}).get(target_direction, 0.3)


def score_baseline_for_direction(
    candidate: dict[str, Any],
    target_direction: str,
    target_strategy: str = "exploration",
    weights: dict[str, float] | None = None,
) -> float:
    """Score a historical round as a baseline candidate for a target direction.

    Hard gates:
      - status must be "success"
      - round_type must not be "idle"
      - verification must have passed (passed_cases == total_cases > 0)

    Scoring dimensions (normalized 0-1):
      - speedup_quality: best_speedup / target_speedup (clamped)
      - direction_compatibility: from compute_direction_compatibility
      - strategy_compatibility: bonus if candidate strategy aligns with target strategy
      - recency: exponential decay by round_index
    """
    default_weights = {
        "speedup": 0.40,
        "direction_compatibility": 0.35,
        "strategy_compatibility": 0.15,
        "recency": 0.10,
    }
    w = weights or default_weights

    # Hard gates
    if candidate.get("status") != "success":
        return 0.0
    if candidate.get("round_type") == "idle":
        return 0.0
    perf_data = candidate.get("perf_data") or {}
    total = perf_data.get("total_cases", 0)
    passed = perf_data.get("passed_cases", 0)
    if not (isinstance(total, (int, float)) and isinstance(passed, (int, float))):
        return 0.0
    if int(total) <= 0 or int(passed) != int(total):
        return 0.0

    best_speedup = candidate.get("best_speedup", 0.0)
    target_speedup = candidate.get("target_speedup", 5.0)
    if not isinstance(best_speedup, (int, float)) or best_speedup <= 0:
        return 0.0

    # Speedup quality (clamped to target, but allow slight overshoot to count).
    speedup_quality = min(best_speedup / max(target_speedup, 1.0), 1.2) / 1.2

    # Direction compatibility.
    baseline_direction = candidate.get("direction", "")
    direction_compat = compute_direction_compatibility(baseline_direction, target_direction)

    # Strategy compatibility.
    baseline_strategy = candidate.get("round_strategy", "exploration")
    strategy_compat = 1.0 if baseline_strategy == target_strategy else 0.5
    # structural_change baselines are generally more useful for other structural directions.
    if baseline_strategy == "structural_change" and target_strategy != "exploration":
        strategy_compat = max(strategy_compat, 0.7)

    # Recency: newer rounds are preferred as tie-breakers. Invert the natural
    # decay so that higher round indices receive higher scores.
    round_index = candidate.get("round_index", 1)
    recency = 1.0 - 1.0 / (1.0 + 0.05 * (round_index - 1))

    score = (
        w.get("speedup", 0.4) * speedup_quality
        + w.get("direction_compatibility", 0.35) * direction_compat
        + w.get("strategy_compatibility", 0.15) * strategy_compat
        + w.get("recency", 0.10) * recency
    )
    return round(max(score, 0.0), 6)


# ---------------------------------------------------------------------------
# Next direction selection
# ---------------------------------------------------------------------------

def select_next_direction(
    round_index: list[dict[str, Any]],
    op_category: str | None = None,
    current_policy: str = "pattern_entry",
    context_changed: bool = False,
) -> tuple[str, str, str, str, list[str], dict[str, Any]]:
    """Select the next optimization direction.

    Directions are now fully LLM-driven. The optional pattern catalog is
    intentionally ignored so that the sub-agent can propose directions based
    on operator analysis rather than a fixed checklist.

    The ``op_category`` argument is kept for interface compatibility but is
    no longer used to select from a fixed pattern catalog.

    Returns:
        (strategy, analysis_policy, direction, hypothesis, evidence_sources, metadata)
    """
    del op_category  # No longer used; LLM-driven directions do not need a catalog.

    policy_rank = {"pattern_entry": 0, "bottleneck_analysis": 1, "ir_supported": 2}
    current_rank = policy_rank.get(current_policy, 0)

    # Natural policy progression: deeper analysis as rounds accumulate.
    policy = current_policy
    round_count = len(round_index)
    if round_count >= 6 and current_rank < 2:
        policy = "ir_supported"
    elif round_count >= 3 and current_rank < 1:
        policy = "bottleneck_analysis"

    if context_changed and policy_rank.get(policy, 0) < 2:
        policy = "bottleneck_analysis" if policy_rank.get(policy, 0) == 0 else "ir_supported"

    if policy == "ir_supported":
        strategy = "structural_change"
    elif policy == "bottleneck_analysis":
        strategy = "exploration"
    else:
        strategy = "exploration"

    hypothesis = (
        "No predefined pattern catalog is enforced. Analyze the operator's "
        "access pattern, bottleneck history, and Triton/Ascend best practices, "
        "then propose and implement the most promising optimization direction."
    )
    evidence_sources = ["model_knowledge"]
    if round_index:
        evidence_sources.append("direction_history")

    metadata = {
        "llm_driven": True,
        "policy_progression": {
            "previous_policy": current_policy,
            "selected_policy": policy,
        },
    }

    return strategy, policy, "pattern: custom", hypothesis, evidence_sources, metadata


def record_switch_manifest(
    switch_dir: Path,
    source_round_dir: str,
    switch_index: int,
    reason: str,
    previous_direction: str,
    new_direction: str,
) -> None:
    """Write SWITCH_MANIFEST.json inside a switch directory."""
    manifest = {
        "source_round_dir": source_round_dir,
        "switch_index": switch_index,
        "reason": reason,
        "previous_direction": previous_direction,
        "new_direction": new_direction,
        "switched_at": datetime.now(timezone.utc).isoformat(),
    }
    (switch_dir / "SWITCH_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Helpers for orchestrator state
# ---------------------------------------------------------------------------

def build_direction_history(
    round_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a compact direction_history array from round_index."""
    history: dict[str, dict[str, Any]] = {}
    for entry in round_index:
        direction = entry.get("direction", "")
        if not direction:
            continue
        if direction not in history:
            history[direction] = {
                "direction": direction,
                "rounds": [],
                "latest_yield": None,
                "best_speedup": 0.0,
            }
        history[direction]["rounds"].append(entry.get("round_dir", ""))
        speedup = entry.get("best_speedup", 0.0)
        if isinstance(speedup, (int, float)) and speedup > history[direction]["best_speedup"]:
            history[direction]["best_speedup"] = speedup
        history[direction]["latest_yield"] = entry.get("direction_yield")
    return list(history.values())
