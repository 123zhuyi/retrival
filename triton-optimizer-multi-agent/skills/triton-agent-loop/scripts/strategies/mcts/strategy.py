#!/usr/bin/env python3
"""MCTS strategy for optimization baseline/round selection.

This strategy wraps the MCTS tree implementation and adapts it to the plugin's
Strategy interface. The Orchestrator drives the round loop; MCTS is only
responsible for:

1. Selecting which historical round (tree node) to expand from.
2. Recording the outcome of completed rounds back into the tree.

MCTS does NOT select optimization directions. Directions are left to the
sub-agent, which receives the MCTS selection path and operator experience as
context.

State is persisted in ``triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.strategy_state/``.
"""

import json
from pathlib import Path
from typing import Any

import direction_selector as ds
from workspace_utils import find_op_name
from ..base import (
    BaselineProposal,
    RoundDecision,
    Strategy,
    StrategyCapabilities,
)
from .mcts_tree import MCTSTree, NodeStatus, ErrorType


_COMPILE_KEYWORDS = (
    "CompilationError",
    "CompileError",
    "SyntaxError",
    "ImportError",
    "ModuleNotFoundError",
    "IndentationError",
    "TypeError",
    "AttributeError",
)


def _tree_state_dir(op_dir: Path) -> Path:
    return op_dir / ".strategy_state"


def _tree_handle(op_dir: Path, op_name: str) -> Path:
    return _tree_state_dir(op_dir) / f"{op_name}_mcts_tree.json"


def _load_or_create_tree(op_dir: Path, op_name: str, params: dict[str, Any]) -> MCTSTree:
    handle = _tree_handle(op_dir, op_name)
    if handle.is_file():
        return MCTSTree.load(str(handle))
    return MCTSTree.create(
        task_id=op_name,
        task_goal=f"Optimize {op_name} via MCTS",
        output_dir=str(_tree_state_dir(op_dir)),
        params=params,
    )


def _load_summary(round_dir: Path) -> dict[str, Any]:
    summary_path = round_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state_json(state_dir: Path, node_id: str, suffix: str, data: dict[str, Any]) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{node_id}_{suffix}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _is_compile_failure(failures: list[dict[str, Any]]) -> bool:
    for failure in failures:
        et = failure.get("error_type", "")
        em = failure.get("error_msg", "")
        if et.startswith(_COMPILE_KEYWORDS) or any(kw in em for kw in _COMPILE_KEYWORDS):
            return True
    return False


def _compute_error_type(summary: dict[str, Any]) -> tuple[str, str]:
    """Return (mcts_error_type, human_reason) based on summary.json."""
    status = summary.get("status", "")
    round_type = summary.get("round_type", "")
    if status == "timeout" or round_type == "idle":
        return ErrorType.OUTPUT_ERROR, summary.get("yield_reason", "timeout_or_idle")

    perf_data = summary.get("perf_data") or {}
    total = perf_data.get("total_cases", 0)
    passed = perf_data.get("passed_cases", 0)
    if not (isinstance(total, (int, float)) and isinstance(passed, (int, float))):
        return ErrorType.COMPILE_ERROR, "missing_perf_data"
    if int(total) <= 0:
        return ErrorType.COMPILE_ERROR, "no_verify_cases"
    if int(passed) != int(total):
        failures = perf_data.get("failures", [])
        if _is_compile_failure(failures):
            return ErrorType.COMPILE_ERROR, summary.get("yield_reason", "compile_failure")
        return ErrorType.OUTPUT_ERROR, summary.get("yield_reason", "verify_failed")

    best_speedup = summary.get("best_speedup", 0.0)
    if not isinstance(best_speedup, (int, float)) or best_speedup <= 0:
        return ErrorType.OUTPUT_ERROR, "invalid_speedup"
    return ErrorType.NONE, "success"


class MCTSStrategy(Strategy):
    """Monte-Carlo Tree Search based round-selection strategy."""

    name = "mcts"
    capabilities = StrategyCapabilities(
        can_propose_direction=False,
        can_propose_baseline=True,
        can_propose_multiple_baselines=False,
        can_decide_switch_internally=False,
        can_recommend_stop=True,
        needs_persistent_state=True,
        can_select_round=True,
    )
    # MCTS tree state is persisted at the operator level by the strategy itself;
    # it is NOT copied between round directories.
    state_file_templates: list[str] = []

    def __init__(self, **params: Any) -> None:
        self.params = params

    def _load_tree(self, op_dir: Path) -> MCTSTree:
        op_name = find_op_name(op_dir) or op_dir.name
        return _load_or_create_tree(op_dir, op_name, self.params)

    def init_for_operator(
        self,
        op_dir: Path,
        config: dict[str, Any],
        round_index: list[dict[str, Any]],
    ) -> None:
        """Ensure the MCTS tree exists for this operator."""
        tree = self._load_tree(op_dir)
        tree.save()

    def _record_from_summary(
        self,
        tree: MCTSTree,
        round_dir: Path,
        summary: dict[str, Any],
    ) -> None:
        """Create/evaluate a tree node for a completed round.

        Idempotent: if the round has already been recorded and evaluated, skip.
        """
        op_dir = round_dir.parent
        op_name = find_op_name(op_dir) or op_dir.name
        round_dir_name = round_dir.name

        # Already recorded?
        if round_dir_name in tree.round_dir_map.values():
            existing_node_id = next(
                (nid for nid, rd in tree.round_dir_map.items() if rd == round_dir_name), None
            )
            if existing_node_id and existing_node_id in tree.nodes:
                if tree.nodes[existing_node_id].status != NodeStatus.PENDING:
                    return

        error_type, reason = _compute_error_type(summary)

        perf_data = summary.get("perf_data") or {}
        verify_json: dict[str, Any] = {
            "total_cases": perf_data.get("total_cases", 0),
            "passed_cases": perf_data.get("passed_cases", 0),
            "failures": perf_data.get("failures", []),
        }
        best_speedup = summary.get("best_speedup", 0.0)
        if not isinstance(best_speedup, (int, float)):
            best_speedup = 0.0
        perf_json: dict[str, Any] = {"speedup_vs_torch": best_speedup}

        state_dir = _tree_state_dir(op_dir)

        # Determine parent from pending expansion; default to root.
        pending = tree.pending_expansion
        if pending and not pending.get("realized"):
            parent_id = pending["parent_id"]
        else:
            parent_id = tree.root_id

        if parent_id not in tree.nodes:
            parent_id = tree.root_id

        description = summary.get("direction") or f"mcts expansion from {parent_id}"

        child_info = tree.create_child(
            parent_id,
            description,
            code_attempt=str(round_dir / f"{op_name}_generated.py"),
            node_id=round_dir_name,
        )
        child_id = child_info["node_id"]
        tree.round_dir_map[child_id] = round_dir_name

        verify_path = _write_state_json(state_dir, child_id, "verify", verify_json)
        perf_path = _write_state_json(state_dir, child_id, "perf", perf_json)
        tree.evaluate(child_id, str(verify_path), str(perf_path))

        if error_type != ErrorType.NONE:
            parent_info = tree.nodes[parent_id].description or parent_id
            error_msg = summary.get("yield_reason", reason)
            tree.record_search_memory(
                error_type=error_type,
                parent_info=parent_info,
                error_msg=error_msg,
                direction=description,
            )

        if pending and not pending.get("realized"):
            pending["realized"] = True
        tree.pending_expansion = None
        tree.save()

    def record_round_result(
        self,
        round_index_entry: dict[str, Any],
        round_dir: Path,
        context: dict[str, Any] | None = None,
    ) -> None:
        op_dir = self._op_dir_from_context(context)
        if op_dir is None:
            return
        summary = _load_summary(round_dir)
        for key in ("direction", "best_speedup", "status", "round_type", "perf_data"):
            if key in round_index_entry and key not in summary:
                summary[key] = round_index_entry[key]
        tree = self._load_tree(op_dir)
        self._record_from_summary(tree, round_dir, summary)

    def _op_dir_from_context(self, context: dict[str, Any] | None) -> Path | None:
        if context is None:
            return None
        op_dir = context.get("op_dir")
        if op_dir is None:
            return None
        return Path(op_dir).expanduser().resolve()

    def select_next_round(
        self,
        round_index: list[dict[str, Any]],
        op_category: str | None,
        current_policy: str,
        context: dict[str, Any] | None = None,
    ) -> RoundDecision:
        """Choose the next round for the MCTS strategy.

        MCTS selects an existing node (round or virtual node) to expand from.
        The actual optimization direction is decided by the sub-agent.
        """
        op_dir = self._op_dir_from_context(context)
        if op_dir is None:
            raise ValueError("MCTSStrategy.select_next_round requires context['op_dir']")

        tree = self._load_tree(op_dir)

        # Reuse an existing pending expansion if it has not been realized yet.
        pending = tree.pending_expansion
        if pending and not pending.get("realized"):
            parent_id = pending["parent_id"]
            parent_round_dir = self._map_parent_to_round_dir(tree, parent_id)
            selection_path = tree.get_path_to_node(parent_id)
            tree.save()
            return RoundDecision(
                direction=None,
                hypothesis=self._build_hypothesis(tree, parent_id),
                baseline=BaselineProposal(mode="single", round_dirs=[parent_round_dir]),
                parent_round_dir=parent_round_dir,
                evidence_sources=["mcts_tree"],
                metadata={
                    "mcts_parent_id": parent_id,
                    "mcts_selection_path": selection_path,
                    "mcts_pending_expansion": True,
                },
            )

        selection = tree.select()
        if selection["action"] == "none":
            return RoundDecision(
                algorithm_recommends_stop=True,
                stop_reason="mcts_tree_fully_explored",
            )

        parent_id = selection["node_id"]
        parent_round_dir = self._map_parent_to_round_dir(tree, parent_id)
        selection_path = tree.get_path_to_node(parent_id)

        tree.pending_expansion = {
            "parent_id": parent_id,
            "realized": False,
        }
        tree.save()

        return RoundDecision(
            direction=None,
            hypothesis=self._build_hypothesis(tree, parent_id),
            baseline=BaselineProposal(mode="single", round_dirs=[parent_round_dir]),
            parent_round_dir=parent_round_dir,
            evidence_sources=["mcts_tree"],
            metadata={
                "mcts_parent_id": parent_id,
                "mcts_selection_path": selection_path,
                "mcts_pending_expansion": True,
            },
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
        meta = dict(decision.metadata)
        if decision.baseline.mode == "single":
            meta["preferred_baseline_dir"] = decision.baseline.round_dirs[0] if decision.baseline.round_dirs else ""
        else:
            meta["preferred_baseline_dir"] = ""
        return (
            decision.round_strategy or "exploration",
            decision.analysis_policy or current_policy,
            decision.direction or "",
            decision.hypothesis or "",
            decision.evidence_sources,
            meta,
        )

    def _map_parent_to_round_dir(self, tree: MCTSTree, parent_id: str) -> str:
        """Map a tree node id to the round directory it represents.

        Root or unmapped virtual nodes map to the empty string, meaning the
        Orchestrator should fall back to the global baseline.
        """
        if parent_id == tree.root_id:
            return ""
        return tree.round_dir_map.get(parent_id, "")

    def _build_hypothesis(self, tree: MCTSTree, parent_id: str) -> str:
        parent_node = tree.nodes.get(parent_id)
        path = tree.get_path_to_node(parent_id)
        if parent_node is None or parent_id == tree.root_id:
            return "MCTS selects the virtual root for a new expansion."
        return (
            f"MCTS selects to expand from node {parent_id} "
            f"(mean reward {parent_node.mean_reward:.4f}, visits {parent_node.visit_count}); "
            f"selection path depth {len(path)}."
        )

    def score_baseline_for_direction(
        self,
        candidate: dict[str, Any],
        target_direction: str,
        target_strategy: str = "exploration",
        context: dict[str, Any] | None = None,
    ) -> float:
        """MCTS normally proposes its own baseline; this is a fallback scorer."""
        op_dir = self._op_dir_from_context(context)
        if op_dir is None:
            return 0.0

        tree = self._load_tree(op_dir)
        pending = tree.pending_expansion
        if not pending or pending.get("realized"):
            return 0.0

        preferred_dir = self._map_parent_to_round_dir(tree, pending["parent_id"])
        if preferred_dir and candidate.get("round_dir") == preferred_dir:
            return 1.0
        return 0.0

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
        # Ensure the tree records this round if called with a round directory.
        op_dir = self._op_dir_from_context(context)
        round_dir = context.get("round_dir") if context else None
        if op_dir is not None and round_dir is not None:
            round_dir_path = Path(round_dir).expanduser().resolve()
            summary = _load_summary(round_dir_path)
            tree = self._load_tree(op_dir)
            self._record_from_summary(tree, round_dir_path, summary)

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
