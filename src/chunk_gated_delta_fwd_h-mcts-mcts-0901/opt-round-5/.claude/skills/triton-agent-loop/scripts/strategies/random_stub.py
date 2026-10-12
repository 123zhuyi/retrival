#!/usr/bin/env python3
"""Random stub strategy: proves the pluggability framework.

This strategy is intentionally simple and non-deterministic. It always returns
a model-driven direction (pattern: custom) so that the sub-agent can autonomously
choose the concrete optimization direction. It is NOT suitable for production
optimization.
"""

import random
from pathlib import Path
from typing import Any

import direction_selector as ds
from .base import (
    BaselineProposal,
    RoundDecision,
    Strategy,
    StrategyCapabilities,
)


class RandomStubStrategy(Strategy):
    """Random direction/baseline selection stub for framework testing."""

    name = "random_stub"
    capabilities = StrategyCapabilities(
        can_propose_direction=True,
        can_propose_baseline=False,
        can_decide_switch_internally=False,
        can_recommend_stop=False,
    )

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_next_round(
        self,
        round_index: list[dict[str, Any]],
        op_category: str | None,
        current_policy: str,
        context: dict[str, Any] | None = None,
    ) -> RoundDecision:
        """Return a model-driven direction placeholder."""
        return RoundDecision(
            direction="pattern: custom",
            hypothesis="LLM-driven autonomous direction selection (catalog absent)",
            round_strategy="exploration",
            analysis_policy=current_policy,
            evidence_sources=["model_knowledge"],
            baseline=BaselineProposal(mode="none"),
            metadata={"stub": True, "llm_driven": True},
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
        decision = self.select_next_round(
            round_index,
            op_category,
            current_policy,
            context,
        )
        return (
            decision.round_strategy or "exploration",
            decision.analysis_policy or current_policy,
            decision.direction or "pattern: autotune",
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
        # Same hard gates as the naive strategy to keep semantics safe.
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
        return self.rng.random()

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
        # Yield classification is not the focus of this stub; delegate to naive.
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
