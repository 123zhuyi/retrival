#!/usr/bin/env python3
"""Abstract base class for optimization direction/baseline strategies.

A strategy encapsulates three decision points of the optimization loop:
1. Which direction to try next (select_next_direction / select_next_round).
2. Which historical round to use as baseline (score_baseline_for_direction).
3. How to label the outcome of a direction attempt (classify_direction_yield).

New algorithms are added by subclassing Strategy and placing the module in
this package. The loader discovers them automatically.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class StrategyCapabilities:
    """Declare what decisions a strategy is able to produce."""

    can_propose_direction: bool = True
    can_propose_baseline: bool = False
    can_propose_multiple_baselines: bool = False
    can_decide_switch_internally: bool = False
    can_recommend_stop: bool = False
    needs_persistent_state: bool = False
    can_select_round: bool = False


@dataclass
class BaselineProposal:
    """Describe which historical round(s) should seed the next round."""

    mode: Literal["none", "single", "multiple"] = "none"
    round_dirs: list[str] = field(default_factory=list)
    combination_rule: str | None = None
    global_baseline_dir: str | None = None


@dataclass
class SwitchMarker:
    """Audit metadata for a direction switch."""

    reason: str
    previous_direction: str | None = None
    new_direction: str | None = None


@dataclass
class RoundDecision:
    """Decision envelope produced by a strategy for the next optimization round.

    Strategies fill only the fields they are capable of determining. The
    Orchestrator completes any missing fields before creating the next round.
    """

    direction: str | None = None
    hypothesis: str | None = None
    round_strategy: str | None = None
    analysis_policy: str | None = None
    evidence_sources: list[str] = field(default_factory=list)
    baseline: BaselineProposal = field(default_factory=BaselineProposal)
    switch_marker: SwitchMarker | None = None
    algorithm_recommends_stop: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_round_dir: str | None = None
    selected_round_dir: str | None = None


class Strategy(ABC):
    """Abstract base class for optimization direction/baseline strategies."""

    # Human-readable identifier. Must be unique across all discovered strategies.
    name: str = ""

    # Capability declaration. Orchestrator uses this to decide which fields of
    # RoundDecision it must fill in.
    capabilities: StrategyCapabilities = StrategyCapabilities()

    # Algorithm-specific state files that must be inherited across rounds.
    # Each entry is a template string relative to the round directory and may
    # contain the {op_name} placeholder. Example: ["gene_pool.json"].
    state_file_templates: list[str] = []

    @abstractmethod
    def select_next_round(
        self,
        round_index: list[dict[str, Any]],
        op_category: str | None,
        current_policy: str,
        context: dict[str, Any] | None = None,
    ) -> RoundDecision:
        """Choose the next optimization round.

        Args:
            round_index: Historical round metadata from round_index.json.
            op_category: Optional operator category (e.g., "matmul", "softmax").
            current_policy: Current analysis_policy (pattern_entry, etc.).
            context: Optional caller context. May include ``op_dir``,
                ``current_round``, or other runtime information strategies can
                use for state management.

        Returns:
            A RoundDecision envelope. Missing fields will be completed by the
            Orchestrator.
        """

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

        New strategies should implement select_next_round() instead. The default
        implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            "select_next_direction is deprecated; implement select_next_round()"
        )

    @abstractmethod
    def score_baseline_for_direction(
        self,
        candidate: dict[str, Any],
        target_direction: str,
        target_strategy: str = "exploration",
        context: dict[str, Any] | None = None,
    ) -> float:
        """Score a historical round as a baseline candidate.

        Returns a value in [0.0, 1.0]. 0.0 means the candidate is ineligible.
        """

    @abstractmethod
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
        """Classify the outcome of a direction attempt.

        Returns:
            (direction_yield, yield_reason, absolute_gain)
        """

    def init_for_operator(
        self,
        op_dir: Path,
        config: dict[str, Any],
        round_index: list[dict[str, Any]],
    ) -> None:
        """Optional lifecycle hook called once per operator workspace.

        Strategies that need per-operator initialization (e.g., creating an
        MCTS tree or loading persistent state) can override this method.
        The default implementation does nothing.
        """
        return None

    def record_round_result(
        self,
        round_index_entry: dict[str, Any],
        round_dir: Path,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Optional lifecycle hook called after a round has finished.

        Strategies that maintain persistent state across rounds (e.g. MCTS
        trees, genetic populations) can use this to record the outcome of the
        just-completed round. The default implementation does nothing.

        Args:
            round_index_entry: The entry from round_index.json for the completed
                round. Contains fields such as direction, best_speedup, status,
                baseline_dir, etc.
            round_dir: Path to the completed round directory (e.g. opt-round-1).
            context: Optional caller context, may include op_dir and config.
        """
        return None

    def resolve_to_single_baseline(
        self,
        baseline: BaselineProposal,
        round_index: list[dict[str, Any]],
        target_direction: str,
        context: dict[str, Any] | None = None,
    ) -> BaselineProposal:
        """Optional hook for strategies that can return multiple baselines.

        If the worker does not support multiple baselines, the Orchestrator
        calls this method to reduce the proposal to a single round_dir.
        The default implementation picks the first entry.
        """
        if baseline.mode == "multiple" and baseline.round_dirs:
            return BaselineProposal(mode="single", round_dirs=[baseline.round_dirs[0]])
        return BaselineProposal(mode="none")
