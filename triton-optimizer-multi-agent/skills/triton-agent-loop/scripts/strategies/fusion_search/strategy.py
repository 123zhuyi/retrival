#!/usr/bin/env python3
"""Fusion-search strategy.

The fusion-search algorithm delegates the bulk of its work to the sub-agent:
- Round 1 performs B1 compute-atom decomposition + B2 fusion analysis and
  spawns parallel routes internally.
- Round 2+ evolves the gene pool internally.

This Orchestrator-side strategy is therefore thin: it only needs to propose a
baseline (global baseline for round 1, best historical round afterwards) and
classify the yield of each completed round.
"""

from pathlib import Path
from typing import Any

import direction_selector as ds
from ..base import (
    BaselineProposal,
    RoundDecision,
    Strategy,
    StrategyCapabilities,
)


class FusionSearchStrategy(Strategy):
    """Strategy wrapper for the fusion-search algorithm."""

    name = "fusion-search"
    capabilities = StrategyCapabilities(
        can_propose_direction=True,
        can_propose_baseline=True,
        can_propose_multiple_baselines=False,
        can_decide_switch_internally=False,
        can_recommend_stop=False,
        needs_persistent_state=False,
        can_select_round=False,
    )
    state_file_templates: list[str] = ["gene_pool.json", "fusion_plan.json"]

    def __init__(self, **params: Any) -> None:
        # Store algorithm parameters (max_routes, crossover_rate, etc.) for
        # potential use by the Orchestrator when building the manifest.
        self.params = params

    def _best_historical_round(
        self, round_index: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Return the highest-speedup round that passed verification."""
        candidates = [
            r
            for r in round_index
            if r.get("best_speedup", 0.0) > 0
            and r.get("direction_yield") != "failed"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: float(r.get("best_speedup", 0.0)))

    def select_next_round(
        self,
        round_index: list[dict[str, Any]],
        op_category: str | None,
        current_policy: str,
        context: dict[str, Any] | None = None,
    ) -> RoundDecision:
        """Choose the next fusion-search round.

        Round 1 starts from the global baseline.  Subsequent rounds start from
        the best verified historical round.
        """
        if not round_index:
            return RoundDecision(
                direction="pattern: custom",
                hypothesis=(
                    "Fusion-search Round 1: decompose compute atoms, perform "
                    "automatic fusion analysis, and build the initial gene pool."
                ),
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                evidence_sources=["model_knowledge"],
                baseline=BaselineProposal(mode="none"),
                metadata={"algorithm_params": self.params},
            )

        best = self._best_historical_round(round_index)
        if best is not None:
            return RoundDecision(
                direction="pattern: custom",
                hypothesis=(
                    "Fusion-search evolutionary round: evolve the gene pool "
                    "from the best historical round."
                ),
                round_strategy="exploration",
                analysis_policy="pattern_entry",
                evidence_sources=["direction_history", "model_knowledge"],
                baseline=BaselineProposal(
                    mode="single", round_dirs=[best.get("round_dir", "")]
                ),
                metadata={"algorithm_params": self.params},
            )

        # No successful historical round yet; fall back to global baseline.
        return RoundDecision(
            direction="pattern: custom",
            hypothesis=(
                "Fusion-search round: continue exploring from the global baseline."
            ),
            round_strategy="exploration",
            analysis_policy="pattern_entry",
            evidence_sources=["model_knowledge"],
            baseline=BaselineProposal(mode="none"),
            metadata={"algorithm_params": self.params},
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
        """Legacy interface preserved during migration."""
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
