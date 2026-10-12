#!/usr/bin/env python3
"""Naive strategy: LLM-driven direction adjustment with internal plateau/low-yield handling.

This strategy is responsible for:
- Letting the sub-agent/LLM propose the concrete optimization direction (always
  surfaced as ``pattern: custom``).
- Detecting when recent rounds are stagnating or yielding low/failed results and
  requesting an adjustment (deeper analysis, different baseline, or a fresh
  optimization idea) instead of stopping the loop.
- Scoring historical baselines for a chosen direction.
- Classifying direction yield after each round.

Stopping is intentionally left to the Orchestrator's C1/C2 checks only; this
strategy no longer recommends stopping.
"""

from pathlib import Path
from typing import Any

import check_local_optimum
import direction_selector as ds
from ..base import (
    BaselineProposal,
    RoundDecision,
    Strategy,
    StrategyCapabilities,
    SwitchMarker,
)


class NaiveStrategy(Strategy):
    """Default strategy: LLM-driven direction selection + low-yield adjustment."""

    name = "naive"
    capabilities = StrategyCapabilities(
        can_propose_direction=True,
        can_propose_baseline=False,
        can_propose_multiple_baselines=False,
        can_decide_switch_internally=True,
        can_recommend_stop=False,
        needs_persistent_state=False,
        can_select_round=False,
    )
    state_file_templates: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        # Naive has no tunable parameters; accept and ignore any extras.
        pass

    def _current_direction(self, round_index: list[dict[str, Any]]) -> str | None:
        if not round_index:
            return None
        direction = round_index[-1].get("direction", "")
        return direction if direction else None

    def _recent_custom_low_yield(
        self,
        round_index: list[dict[str, Any]],
        window: int = 3,
    ) -> bool:
        """Return True if the last ``window`` non-idle rounds are low-yield or failed.

        Directions are now always surfaced as ``pattern: custom`` by the
        LLM-driven selector, so the check only looks at the recent yield trend.
        When the same open-ended direction keeps producing poor results, the
        strategy asks for an adjustment rather than giving up.
        """
        recent = [r for r in round_index if r.get("round_type") != "idle"][-window:]
        if len(recent) < window:
            return False
        for entry in recent:
            yield_label = entry.get("direction_yield", "")
            if yield_label not in ("low", "failed"):
                return False
        return True

    def select_next_round(
        self,
        round_index: list[dict[str, Any]],
        op_category: str | None,
        current_policy: str,
        context: dict[str, Any] | None = None,
    ) -> RoundDecision:
        """Choose the next round for the naive strategy.

        The naive strategy now only owns the "adjustment" decision:
          - If recent rounds show a speedup plateau (C3), request a direction
            adjustment by escalating the analysis policy.
          - If recent ``pattern: custom`` attempts are consistently low-yield or
            failed, also request an adjustment.
          - Otherwise continue normal exploration.

        The strategy never recommends stopping; the Orchestrator handles C1/C2.
        """
        op_dir = Path(context["op_dir"]) if context and context.get("op_dir") else None
        current_direction = self._current_direction(round_index)

        # C3: local optimum / plateau detected from disk speedups.
        plateau_detected = False
        if op_dir is not None:
            try:
                local_opt = check_local_optimum.check_local_optimum(
                    op_dir, window=3, max_gain=0.02
                )
                plateau_detected = local_opt.get("in_local_optimum", False)
            except Exception:
                plateau_detected = False

        # LLM-driven low-yield adjustment: consecutive custom directions that are
        # low or failed also trigger a context change / deeper analysis.
        low_yield_adjustment = self._recent_custom_low_yield(round_index, window=3)

        if plateau_detected or low_yield_adjustment:
            switch_reason = (
                "plateau_stagnation" if plateau_detected else "low_yield_adjustment"
            )
            strategy, policy, direction, hypothesis, evidence, meta = (
                ds.select_next_direction(
                    round_index,
                    op_category=op_category,
                    current_policy=current_policy,
                    context_changed=True,
                )
            )
            return RoundDecision(
                direction=direction,
                hypothesis=(
                    f"Recent rounds show {'a speedup plateau' if plateau_detected else 'consistent low/failed yields'}. "
                    "Adjust the optimization approach: re-analyze the bottleneck, "
                    "try a different implementation idea, or escalate the analysis depth."
                ),
                round_strategy=strategy,
                analysis_policy=policy,
                evidence_sources=evidence,
                baseline=BaselineProposal(mode="none"),
                switch_marker=SwitchMarker(
                    reason=switch_reason,
                    previous_direction=current_direction,
                    new_direction=direction,
                ),
                metadata=meta,
            )

        # Normal path: let the sub-agent/LLM continue with the current policy.
        strategy, policy, direction, hypothesis, evidence, meta = (
            ds.select_next_direction(
                round_index,
                op_category=op_category,
                current_policy=current_policy,
                context_changed=False,
            )
        )
        return RoundDecision(
            direction=direction,
            hypothesis=hypothesis,
            round_strategy=strategy,
            analysis_policy=policy,
            evidence_sources=evidence,
            baseline=BaselineProposal(mode="none"),
            metadata=meta,
        )

    def select_next_direction(
        self,
        round_index: list[dict[str, Any]],
        op_category: str | None,
        current_policy: str,
        context_changed: bool = False,
        direction_exhausted: bool = False,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str, str, str, list[str], dict[str, Any]]:
        """Legacy interface preserved during migration.

        New code should call select_next_round() instead.
        """
        decision = self.select_next_round(
            round_index,
            op_category,
            current_policy,
            context,
        )
        return (
            decision.round_strategy or "exploration",
            decision.analysis_policy or current_policy,
            decision.direction or "pattern: custom",
            decision.hypothesis or "",
            decision.evidence_sources,
            decision.metadata,
        )

    def score_baseline_for_direction(
        self,
        candidate: dict[str, Any],
        target_direction: str,
        target_strategy: str = "exploration",
        context: dict[str, Any] | None = None,
    ) -> float:
        return ds.score_baseline_for_direction(
            candidate,
            target_direction,
            target_strategy=target_strategy,
        )

    def classify_direction_yield(
        self,
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
    ) -> tuple[str, str, float]:
        return ds.classify_direction_yield(
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
