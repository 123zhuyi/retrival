#!/usr/bin/env python3
"""orchestrator_loop.py — 主 Orchestrator 的 round 调度循环参考实现。

用法:
    python3 orchestrator_loop.py \
        --op-name layer_norm \
        --algorithm <discovered-algorithm> \
        --run-tag default \
        --mode A \
        --task-desc path/to/layer_norm.py \
        [--task-json path/to/layer_norm.json] \
        [--gpu-kernel-ref path/to/gpu_kernel_ref.py] \
        [--gpu-perf-csv path/to/gpu_perf.csv] \
        [--timeout 7200] \
        [--op-category matmul]

    可用算法由 skills/triton-agent-loop/scripts/strategies/ 下的 Strategy 子类动态发现。

说明:
    本脚本仅作为参考实现。主 Claude Code Agent 可以直接调用其中的函数，
    也可以复现其逻辑。它负责：
    1. 初始化 workspace 和 state-{op_name}-{algorithm}-{run_tag}.json
    2. 循环生成 task_manifest.json
    3. 调用 dispatch_round.py 启动子 CLI
    4. 解析 round_result.json
    5. 执行 C1/C2 停止/继续判定
    6. 停止时选择全局最优 round 并输出最终报告

工作区隔离:
    不同算法或同一算法的不同运行使用独立目录：
        triton_ascend_output/{op_name}-{algorithm}-{run_tag}/

设计变更（v2）:
    - C3/D4 判定从 Orchestrator 移除，下沉到各进化策略内部。
    - 不再创建 opt-round-N-switch 目录；方向切换只作为元数据记录。
    - 所有策略通过 RoundDecision 信封返回决策，Orchestrator 只补全缺失项。
    - algorithm 不再配置在 config.json 中；由用户在会话开始时选择，并写入 state。
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config_loader import (
    read_config,
    get_enabled_algorithms,
    get_algorithm_params,
)
from direction_selector import build_direction_history
from strategies import discover_strategies, load_strategy_for_algorithm
from strategies.base import BaselineProposal, RoundDecision


SCRIPT_DIR = Path(__file__).resolve().parent
CTRL_SKILLS_DIR = SCRIPT_DIR.parent
REPO_OPS_CANDIDATES = [
    SCRIPT_DIR.parents[5] / "ops",
    SCRIPT_DIR.parents[6] / "ops",
]


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def find_ops_root() -> Path | None:
    """Find shared ops/ directory."""
    for candidate in REPO_OPS_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def _algorithm_reference(algorithm: str) -> str:
    """Return the reference document path for an algorithm."""
    return f".claude/references/algorithms/{algorithm}/round.md"


def select_algorithm(
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> str | None:
    """Determine the active optimization algorithm for this task.

    Priority:
      1. ``state["algorithm"]`` if present and still enabled.
      2. If exactly one algorithm is enabled in ``config["algorithms"]``, use it.
      3. Otherwise return ``None`` so the caller can ask the user.

    The main Claude Code agent should use ``AskUserQuestion`` when this function
    returns ``None``.  When running this script directly, pass ``--algorithm``.
    """
    discovered = set(discover_strategies().keys())
    enabled = [a for a in get_enabled_algorithms(config) if a in discovered]
    if not enabled:
        raise ValueError("No algorithms are enabled in config.json")

    if state and isinstance(state.get("algorithm"), str):
        algo = state["algorithm"]
        if algo in enabled:
            return algo

    if len(enabled) == 1:
        return enabled[0]

    return None


def _discovered_algorithm_names() -> set[str]:
    """Return the set of algorithm names discovered from strategy modules."""
    return set(discover_strategies().keys())


def _find_legacy_workspace(
    cwd: Path, op_name: str
) -> tuple[Path, dict[str, Any]] | None:
    """Detect a legacy workspace: triton_ascend_output/{op_name}/ + state-{op_name}.json."""
    legacy_op_dir = cwd / "triton_ascend_output" / op_name
    legacy_state_path = cwd / ".triton-agent" / f"state-{op_name}.json"
    if not legacy_op_dir.is_dir() or not legacy_state_path.is_file():
        return None
    try:
        return legacy_op_dir, load_json(legacy_state_path)
    except ValueError:
        return None


def _infer_algorithm_from_legacy(
    legacy_state: dict[str, Any],
    legacy_op_dir: Path,
    discovered: set[str],
) -> str | None:
    """Infer the algorithm used by a legacy workspace from multiple sources."""
    state_algo = legacy_state.get("algorithm")
    if isinstance(state_algo, str) and state_algo in discovered:
        return state_algo

    round_index_path = legacy_op_dir / "round_index.json"
    if round_index_path.is_file():
        try:
            data = load_json(round_index_path)
            if isinstance(data, list) and data:
                algo = data[0].get("algorithm")
                if isinstance(algo, str) and algo in discovered:
                    return algo
        except ValueError:
            pass

    for summary_path in sorted(legacy_op_dir.glob("opt-round-*/summary.json")):
        try:
            algo = load_json(summary_path).get("algorithm")
            if isinstance(algo, str) and algo in discovered:
                return algo
        except ValueError:
            pass

    return None


def _discover_algorithm_from_new_state(cwd: Path, op_name: str) -> str | None:
    """Scan new-format state files and read the saved algorithm.

    Returns the algorithm if exactly one matching state file exists; otherwise
    returns None so the caller can ask the user or fall back to other sources.
    """
    state_dir = cwd / ".triton-agent"
    if not state_dir.is_dir():
        return None
    matches = list(state_dir.glob(f"state-{op_name}-*.json"))
    if len(matches) != 1:
        return None
    try:
        return load_json(matches[0]).get("algorithm")
    except ValueError:
        return None


def _list_new_state_algorithms(
    cwd: Path,
    op_name: str,
    run_tag: str | None = None,
) -> dict[str, Path]:
    """Return a mapping of algorithm -> state file path for new-format states.

    If *run_tag* is provided, only state files whose names end with
    ``-{run_tag}.json`` are considered. This isolates workspaces that differ
    only by run tag so that ``--algorithm naive --run-tag run2`` does not
    conflict with an existing ``state-{op_name}-mcts-default.json``.
    """
    state_dir = cwd / ".triton-agent"
    if not state_dir.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in state_dir.glob(f"state-{op_name}-*.json"):
        if run_tag is not None and not path.name.endswith(f"-{run_tag}.json"):
            continue
        try:
            algo = load_json(path).get("algorithm")
            if isinstance(algo, str) and algo:
                result[algo] = path
        except ValueError:
            continue
    return result


def _build_legacy_report(
    op_name: str,
    legacy_op_dir: Path,
    legacy_state: dict[str, Any],
    inferred: str | None,
) -> dict[str, Any]:
    """Build a structured report about a detected legacy workspace."""
    return {
        "type": "legacy_workspace_detected",
        "op_name": op_name,
        "legacy_dir": str(legacy_op_dir),
        "legacy_state": str(
            legacy_op_dir.parent.parent / ".triton-agent" / f"state-{op_name}.json"
        ),
        "current_round": legacy_state.get("current_round", "unknown"),
        "last_work_dir": legacy_state.get("work_dir", "unknown"),
        "state_algorithm": legacy_state.get("algorithm", "unknown"),
        "inferred_algorithm": inferred,
        "migration_plan": (
            f"Rename {legacy_op_dir.name} to {op_name}-<algorithm>-<run_tag>/, "
            f"create .triton-agent/state-{op_name}-<algorithm>-<run_tag>.json, "
            "preserve all round data."
        ),
        "options": [
            {
                "id": "continue_inferred",
                "label": f"Continue with inferred algorithm '{inferred or 'unknown'}'",
            },
            {"id": "specify_algorithm", "label": "Specify a different algorithm"},
            {"id": "abandon", "label": "Abandon legacy workspace and start fresh"},
        ],
    }


def _migrate_legacy_workspace(
    cwd: Path,
    op_name: str,
    algorithm: str,
    run_tag: str,
    legacy_op_dir: Path,
    legacy_state: dict[str, Any],
) -> Path:
    """Migrate a legacy workspace to the new naming scheme.

    Returns the new operator directory path.
    """
    new_op_dir = cwd / "triton_ascend_output" / f"{op_name}-{algorithm}-{run_tag}"
    new_state_path = cwd / ".triton-agent" / f"state-{new_op_dir.name}.json"
    if new_op_dir.exists():
        raise FileExistsError(f"Migration target already exists: {new_op_dir}")

    legacy_op_dir.rename(new_op_dir)

    current_round = legacy_state.get("current_round", 1)
    work_dir = new_op_dir / f"opt-round-{current_round}"
    legacy_state.update(
        {
            "algorithm": algorithm,
            "run_tag": run_tag,
            "algorithm_reference": _algorithm_reference(algorithm),
            "work_dir": str(work_dir),
        }
    )
    save_json(new_state_path, legacy_state)

    # Ensure a round_index.json exists so the migration is treated as a recovery
    # rather than a fresh initialization.
    round_index_path = new_op_dir / "round_index.json"
    if not round_index_path.is_file():
        save_json(round_index_path, [])

    legacy_state_path = cwd / ".triton-agent" / f"state-{op_name}.json"
    if legacy_state_path.is_file():
        legacy_state_path.write_text(
            json.dumps(
                {
                    "migrated": True,
                    "migrated_to": str(new_state_path),
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    return new_op_dir

def init_workspace(
    cwd: Path,
    op_name: str,
    op_dir: Path,
    mode: str,
    task_desc: Path,
    task_json: Path | None,
    gpu_kernel_ref: Path | None,
    gpu_perf_csv: Path | None,
    config: dict[str, Any],
    algorithm: str,
    run_tag: str = "default",
) -> Path:
    """Initialize workspace for Round 1."""
    algorithm_ref = _algorithm_reference(algorithm)

    round_dir = op_dir / "opt-round-1"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Global baseline: copy the user-provided task file as the immutable reference.
    global_baseline_dir = op_dir / "global_baseline"
    global_baseline_dir.mkdir(parents=True, exist_ok=True)
    global_baseline_code = global_baseline_dir / f"{op_name}_generated.py"
    if task_desc.is_file():
        shutil.copy2(task_desc, global_baseline_code)
    baseline_info = {
        "source_path": str(task_desc),
        "baseline_speedup": 1.0,
        "established_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(global_baseline_dir / "baseline_info.json", baseline_info)

    # Operator-level experience file (private to this operator run).
    experience_path = op_dir / "operator_experience.md"
    experience_path.write_text(
        f"# Operator Experience: {op_name} ({algorithm}/{run_tag})\n\n"
        "## Failures / Avoid\n"
        "<!-- Sub-agent appends here after each round -->\n\n"
        "## Successful Patterns\n"
        "<!-- Sub-agent appends here after each round -->\n\n"
        "## Notes\n",
        encoding="utf-8",
    )

    state = {
        "op_name": op_name,
        "algorithm": algorithm,
        "run_tag": run_tag,
        "status": "active",
        "current_round": 1,
        "algorithm_reference": algorithm_ref,
        "rounds": {
            "1": {"status": "active", "round_dir": "opt-round-1"},
        },
        "phase": "round_active",
        "work_dir": str(round_dir),
        "last_phase": 0,
        "performance_history": [],
        "direction_history": [],
    }
    state_dir = cwd / ".triton-agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    save_json(state_dir / f"state-{op_dir.name}.json", state)

    summary = {
        "round_index": 1,
        "algorithm": algorithm,
        "baseline_dir": "global_baseline",
        "global_baseline_dir": "global_baseline",
        "global_baseline_speedup": 1.0,
        "round_strategy": "exploration",
        "analysis_policy": "pattern_entry",
        "hypothesis": f"Initial LLM-driven exploration for {op_name} optimization",
        "direction": "pattern: custom",
        "evidence_sources": ["model_knowledge"],
        "status": "active",
        "best_speedup": 0.0,
        "success": False,
        "target_speedup": config.get("target_speedup", 5),
    }
    save_json(round_dir / "summary.json", summary)

    round_index = [{
        "round_index": 1,
        "round_dir": "opt-round-1",
        "algorithm": algorithm,
        "direction": "pattern: custom",
        "hypothesis": f"Initial LLM-driven exploration for {op_name} optimization",
        "evidence_sources": ["model_knowledge"],
        "analysis_policy": "pattern_entry",
        "effective_metric_source": "kernel",
        "status": "active",
        "best_speedup": 0.0,
        "round_strategy": "exploration",
        "direction_yield": None,
        "yield_reason": None,
        "absolute_gain_vs_baseline": None,
        "absolute_gain_vs_global_baseline": None,
        "baseline_dir": "global_baseline",
        "global_baseline_dir": "global_baseline",
        "global_baseline_speedup": 1.0,
        "consecutive_low_gain_count": 0,
        "round_type": "active",
        "target_speedup": config.get("target_speedup", 5),
    }]
    save_json(op_dir / "round_index.json", round_index)

    return round_dir


def build_manifest(
    op_dir: Path,
    op_name: str,
    round_index: int,
    baseline_dir: Path | None,
    algorithm: str,
    round_strategy: str,
    analysis_policy: str,
    hypothesis: str,
    direction: str,
    evidence_sources: list[str],
    mode: str,
    task_desc: Path,
    task_json: Path | None,
    gpu_kernel_ref: Path | None,
    gpu_perf_csv: Path | None,
    config: dict[str, Any],
    arch: str = "ascend910b1",
    lineage_summary: list[dict[str, Any]] | None = None,
    experience_file: str | None = None,
    global_baseline_dir: Path | None = None,
    global_baseline_speedup: float = 1.0,
    decision_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write task_manifest.json for the round worker."""
    work_dir = op_dir / f"opt-round-{round_index}"
    work_dir.mkdir(parents=True, exist_ok=True)

    algorithm_reference = _algorithm_reference(algorithm)
    algorithm_params = get_algorithm_params(config, algorithm)

    # Optionally surface the parent round's detailed journal to the sub-agent.
    parent_round_journal: str | None = None
    if baseline_dir is not None:
        journal_path = Path(baseline_dir) / "round_journal.md"
        if journal_path.is_file():
            parent_round_journal = str(journal_path)

    manifest: dict[str, Any] = {
        "op_name": op_name,
        "round_index": round_index,
        "work_dir": str(work_dir),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "algorithm": algorithm,
        "algorithm_reference": algorithm_reference,
        "algorithm_params": algorithm_params,
        "round_strategy": round_strategy,
        "analysis_policy": analysis_policy,
        "hypothesis": hypothesis,
        "direction": direction,
        "evidence_sources": evidence_sources,
        "mode": mode,
        "arch": arch,
        "input_files": {
            "task_desc": str(task_desc),
            "task_json": str(task_json) if task_json else None,
            "gpu_kernel_ref": str(gpu_kernel_ref) if gpu_kernel_ref else None,
            "gpu_perf_csv": str(gpu_perf_csv) if gpu_perf_csv else None,
        },
        "config": config,
        "is_recovery": False,
        "worker_reference": algorithm_reference,
        "lineage_summary": lineage_summary or [],
        "experience_file": experience_file,
        "global_baseline_dir": str(global_baseline_dir) if global_baseline_dir else None,
        "global_baseline_speedup": global_baseline_speedup,
        "parent_round_journal": parent_round_journal,
    }

    # Forward algorithm-specific metadata from the strategy decision envelope.
    # For MCTS this includes mcts_parent_id, mcts_selection_path, etc.
    if decision_metadata:
        for key in ("mcts_parent_id", "mcts_selection_path", "mcts_pending_expansion"):
            if key in decision_metadata:
                manifest[key] = decision_metadata[key]

    manifest_path = op_dir / f".task_manifest_{round_index}.json"
    save_json(manifest_path, manifest)
    return manifest_path


def dispatch_round(manifest_path: Path, timeout: int = 7200) -> dict[str, Any]:
    """Call dispatch_round.py to spawn sub CLI."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "dispatch_round.py"),
        "--manifest", str(manifest_path),
        "--timeout", str(timeout),
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return json.loads(output)
    except subprocess.CalledProcessError as exc:
        try:
            return json.loads(exc.output)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "stderr": exc.output,
                "returncode": exc.returncode,
            }
    except json.JSONDecodeError as exc:
        return {"status": "error", "stderr": f"invalid JSON from dispatch_round.py: {exc}"}


def update_round_index(
    op_dir: Path,
    round_result: dict[str, Any],
    active_algorithm: str | None = None,
) -> None:
    """Update round_index.json with completed round info.

    Uses ``active_algorithm`` as the fallback when the sub-agent result does not
    report an algorithm. This prevents a hard-coded "naive" default from causing
    algorithm drift when the active run uses mcts/fusion-search.
    """
    round_index_path = op_dir / "round_index.json"
    round_index = load_json(round_index_path) if round_index_path.is_file() else []
    if not isinstance(round_index, list):
        round_index = []

    idx = round_result.get("round_index", 0)
    entry = next((r for r in round_index if r.get("round_index") == idx), None)
    if entry is None:
        entry = {
            "round_index": idx,
            "round_dir": f"opt-round-{idx}",
        }
        round_index.append(entry)

    entry.update({
        "algorithm": round_result.get("algorithm") or active_algorithm or "naive",
        "direction": round_result.get("direction", ""),
        "hypothesis": round_result.get("hypothesis", ""),
        "effective_metric_source": round_result.get("effective_metric_source", "kernel"),
        "status": round_result.get("status", "failed"),
        "best_speedup": round_result.get("best_speedup", 0.0),
        "round_strategy": round_result.get("round_strategy", "exploration"),
    })

    # Optional fields enriched by the sub-agent; preserve existing values if absent.
    optional_fields = [
        "evidence_sources",
        "analysis_policy",
        "direction_yield",
        "yield_reason",
        "absolute_gain_vs_baseline",
        "baseline_dir",
        "design_summary",
        "iter_count",
        "opt_iter_count",
        "key_fixes",
        "bottlenecks_observed",
        "parent_round_journal",
    ]
    for key in optional_fields:
        if key in round_result:
            entry[key] = round_result[key]

    save_json(round_index_path, round_index)


def check_stop_decision(
    op_dir: Path,
    current_round: int,
    best_speedup: float,
    target_speedup: float,
    max_rounds: int,
    algorithm_recommends_stop: bool = False,
) -> dict[str, Any]:
    """Run check_stop_decision.py."""
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "check_stop_decision.py"),
        "--op-dir", str(op_dir),
        "--current-round", str(current_round),
        "--best-speedup", str(best_speedup),
        "--target-speedup", str(target_speedup),
        "--max-rounds", str(max_rounds),
    ]
    if algorithm_recommends_stop:
        cmd.append("--algorithm-recommends-stop")
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return json.loads(output)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {"can_stop": False, "reason": str(exc)}


def count_non_idle_rounds(round_index: list[dict[str, Any]]) -> int:
    return sum(1 for r in round_index if r.get("status") != "skipped_idle")


def build_lineage_summary(
    round_index: list[dict[str, Any]],
    target_round_dir: str,
) -> list[dict[str, Any]]:
    """Build a root-to-leaf lineage summary for the target round."""
    by_dir = {entry.get("round_dir"): entry for entry in round_index if entry.get("round_dir")}
    lineage: list[dict[str, Any]] = []
    visited: set[str] = set()
    current_dir = target_round_dir

    while current_dir and current_dir not in visited:
        visited.add(current_dir)
        entry = by_dir.get(current_dir)
        if entry is None:
            break
        lineage_entry: dict[str, Any] = {
            "round_index": entry.get("round_index"),
            "round_dir": current_dir,
            "direction": entry.get("direction"),
            "best_speedup": entry.get("best_speedup"),
            "direction_yield": entry.get("direction_yield"),
            "hypothesis": entry.get("hypothesis"),
            "parent_round_dir": entry.get("baseline_dir"),
        }
        # Enrich the summary with design/iteration metadata when available.
        for key in (
            "design_summary",
            "iter_count",
            "opt_iter_count",
            "key_fixes",
            "bottlenecks_observed",
            "outcome",
        ):
            if key in entry:
                lineage_entry[key] = entry[key]
        lineage.append(lineage_entry)
        current_dir = entry.get("baseline_dir") or ""

    lineage.reverse()
    return lineage


def resolve_baseline(
    op_dir: Path,
    decision: RoundDecision,
    round_index: list[dict[str, Any]],
    op_name: str,
) -> Path | None:
    """Resolve the baseline directory from a RoundDecision.

    Returns the path to the baseline generated code, or None if the next round
    should start without a baseline. If the strategy-selected parent round has
    no code (e.g., a virtual MCTS node), fall back to the global baseline.
    """
    baseline = decision.baseline
    if baseline.mode == "none":
        selected_dir = ""
    elif baseline.mode == "single":
        selected_dir = baseline.round_dirs[0] if baseline.round_dirs else ""
    else:
        selected_dir = ""
        for round_dir_name in baseline.round_dirs:
            if not round_dir_name:
                continue
            candidate = op_dir / round_dir_name / f"{op_name}_generated.py"
            if candidate.is_file():
                return candidate

    if selected_dir:
        candidate = op_dir / selected_dir / f"{op_name}_generated.py"
        if candidate.is_file():
            return candidate

    # Fallback to the immutable global baseline (user-provided reference code).
    global_baseline_candidate = op_dir / "global_baseline" / f"{op_name}_generated.py"
    if global_baseline_candidate.is_file():
        return global_baseline_candidate
    return None


def prepare_next_round(
    op_dir: Path,
    current_round: int,
    decision: RoundDecision,
    config: dict[str, Any],
    round_index: list[dict[str, Any]],
    algorithm: str,
    op_category: str | None = None,
    global_baseline_dir: str | None = "global_baseline",
    global_baseline_speedup: float = 1.0,
) -> dict[str, Any]:
    """Prepare the next round directory using transition_next_round.py."""
    switch_marker = decision.switch_marker
    algorithm_metadata = dict(decision.metadata or {})
    algorithm_params = get_algorithm_params(config, algorithm)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "transition_next_round.py"),
        "--op-dir", str(op_dir),
        "--current-round", str(current_round),
        "--next-round-strategy", decision.round_strategy or "exploration",
        "--next-analysis-policy", decision.analysis_policy or "pattern_entry",
        "--next-hypothesis", decision.hypothesis or "",
        "--next-direction", decision.direction or "",
        "--next-evidence-sources", json.dumps(decision.evidence_sources, ensure_ascii=False),
        "--strategy-name", algorithm,
        "--strategy-params", json.dumps(algorithm_params, ensure_ascii=False),
        "--algorithm", algorithm,
        "--algorithm-metadata", json.dumps(algorithm_metadata, ensure_ascii=False),
        "--current-success-criteria", "speedup_vs_baseline >= target",
        "--current-key-learnings", "",
        "--global-baseline-dir", global_baseline_dir or "global_baseline",
        "--global-baseline-speedup", str(global_baseline_speedup),
    ]

    baseline = decision.baseline
    if baseline.mode == "single" and baseline.round_dirs:
        cmd.extend(["--baseline-dir", baseline.round_dirs[0]])
    elif baseline.mode == "multiple":
        cmd.extend(["--baseline-dirs", json.dumps(baseline.round_dirs, ensure_ascii=False)])
        if baseline.combination_rule:
            cmd.extend(["--combination-rule", baseline.combination_rule])

    if switch_marker is not None:
        cmd.extend([
            "--switch-marker",
            json.dumps({
                "reason": switch_marker.reason,
                "previous_direction": switch_marker.previous_direction,
                "new_direction": switch_marker.new_direction,
            }, ensure_ascii=False),
        ])

    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return json.loads(output)


def _update_global_template(
    cwd: Path,
    op_name: str,
    category: str | None,
    best_round_entry: dict[str, Any] | None,
    operator_experience_path: Path,
) -> Path | None:
    """Append or replace the operator-specific experience in the project template.

    The project template lives at ``.claude/template/{category}.md`` and is
    copied per project by ``init.sh``. Updates only affect the current project
    unless manually synced across projects.
    """
    category = category or "general"
    template_dir = cwd / ".claude" / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / f"{category}.md"

    heading = f"## Experience: {op_name}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    best_round = best_round_entry.get("round_index") if best_round_entry else None
    best_speedup = best_round_entry.get("best_speedup") if best_round_entry else None
    target_reached = best_round_entry.get("target_reached") if best_round_entry else None
    final_code = (
        best_round_entry.get("artifacts", {}).get("generated_code")
        if best_round_entry
        else None
    )

    lines = [
        heading,
        "",
        f"- **Updated**: {timestamp}",
        f"- **Best round**: {best_round}",
        f"- **Best speedup**: {best_speedup}",
        f"- **Target reached**: {target_reached}",
    ]
    if final_code:
        lines.append(f"- **Final code**: `{final_code}`")

    if operator_experience_path.is_file():
        exp_text = operator_experience_path.read_text(encoding="utf-8").strip()
        if exp_text:
            lines.extend(["", "### Operator experience", "", exp_text])

    section = "\n".join(lines) + "\n\n"

    if template_path.is_file():
        content = template_path.read_text(encoding="utf-8")
    else:
        content = f"# {category.capitalize()} operator optimization experience\n\n"

    # Replace an existing section for the same operator, or append if not present.
    idx = content.find(f"\n{heading}")
    if idx == -1:
        idx = content.find(heading)
    if idx != -1:
        next_heading = content.find("\n## ", idx + 1)
        if next_heading == -1:
            content = content[:idx] + section
        else:
            content = content[:idx] + section + content[next_heading + 1:]
    else:
        content = content.rstrip() + "\n\n" + section

    template_path.write_text(content, encoding="utf-8")
    return template_path


def run_orchestrator(
    cwd: Path,
    op_name: str,
    mode: str,
    task_desc: Path,
    task_json: Path | None,
    gpu_kernel_ref: Path | None,
    gpu_perf_csv: Path | None,
    timeout: int = 7200,
    op_category: str | None = None,
    arch: str = "ascend910b1",
    algorithm: str | None = None,
    run_tag: str = "default",
    migrate_legacy: str | None = None,
    detect_legacy: bool = False,
) -> dict[str, Any]:
    """Main orchestration loop."""
    config = read_config(cwd)
    target_speedup = config.get("target_speedup", 5)
    max_rounds = config.get("max_rounds", 10)
    max_worker_timeout = config.get("max_worker_timeout", 28800)

    # Determine the active algorithm before constructing the workspace path.
    # Priority:
    #   1. CLI --algorithm (optionally combined with --migrate-legacy)
    #   2. New-format state file (state-{op_name}-{algorithm}-{run_tag}.json)
    #   3. Legacy workspace with user confirmation
    #   4. Config selection (single enabled algorithm)
    #   5. Ask user (returns None; caller must handle)
    discovered = _discovered_algorithm_names()

    # --detect-legacy: only report, do not modify anything.
    if detect_legacy:
        legacy = _find_legacy_workspace(cwd, op_name)
        if legacy is None:
            print(json.dumps({"type": "none"}, indent=2, ensure_ascii=False))
            sys.exit(0)
        legacy_op_dir, legacy_state = legacy
        inferred = _infer_algorithm_from_legacy(legacy_state, legacy_op_dir, discovered)
        report = _build_legacy_report(op_name, legacy_op_dir, legacy_state, inferred)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0)

    # CLI override: validate against discovered strategies.
    if algorithm is not None and algorithm not in discovered:
        raise ValueError(
            f"Algorithm '{algorithm}' not discovered. Available: {sorted(discovered)}"
        )

    # Discover any existing new-format state files first so we can enforce
    # consistency with the CLI override and detect conflicting workspaces.
    existing_state_algorithms = _list_new_state_algorithms(cwd, op_name, run_tag)

    # 1. Resolve algorithm: CLI > existing state > legacy > config selection.
    if algorithm is None:
        if len(existing_state_algorithms) == 1:
            new_state_algorithm = next(iter(existing_state_algorithms.keys()))
            if new_state_algorithm not in discovered:
                raise ValueError(
                    f"Saved algorithm '{new_state_algorithm}' has no strategy. "
                    f"Available: {sorted(discovered)}"
                )
            algorithm = new_state_algorithm
        elif len(existing_state_algorithms) > 1:
            algos = sorted(existing_state_algorithms.keys())
            raise ValueError(
                f"Multiple existing workspaces found for '{op_name}': {algos}. "
                f"Please specify --algorithm or --migrate-legacy."
            )
    else:
        # CLI specified: ensure it does not conflict with an existing workspace.
        for existing_algo, existing_path in existing_state_algorithms.items():
            if existing_algo != algorithm:
                raise ValueError(
                    f"Existing workspace uses algorithm '{existing_algo}' "
                    f"({existing_path.name}). Conflicts with --algorithm {algorithm}."
                )

    # 2. Try legacy workspace inference when no new-format state exists.
    legacy: tuple[Path, dict[str, Any]] | None = None
    if not existing_state_algorithms:
        legacy = _find_legacy_workspace(cwd, op_name)
        if legacy is not None:
            legacy_op_dir, legacy_state = legacy
            inferred = _infer_algorithm_from_legacy(legacy_state, legacy_op_dir, discovered)

            if migrate_legacy:
                algorithm = migrate_legacy
                if inferred and algorithm != inferred:
                    print(
                        f"WARN: requested migration algorithm '{algorithm}' differs from "
                        f"inferred '{inferred}'. Using requested algorithm.",
                        file=sys.stderr,
                    )
            else:
                report = _build_legacy_report(
                    op_name, legacy_op_dir, legacy_state, inferred
                )
                print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
                sys.exit(3)

    # 3. Fall back to config selection.
    if algorithm is None:
        algorithm = select_algorithm(config, None)

    if algorithm is None:
        raise ValueError(
            "No active algorithm selected. In the main Claude Code agent use "
            "AskUserQuestion; when running this script directly pass --algorithm."
        )

    # Algorithm is immutable for this run.
    active_algorithm = algorithm

    op_dir = cwd / "triton_ascend_output" / f"{op_name}-{active_algorithm}-{run_tag}"
    state_path = cwd / ".triton-agent" / f"state-{op_dir.name}.json"

    # Execute legacy migration if a legacy workspace exists and the new-format
    # workspace has not already been created.
    if legacy is not None and not op_dir.exists():
        legacy_op_dir, legacy_state = legacy
        op_dir = _migrate_legacy_workspace(
            cwd, op_name, active_algorithm, run_tag, legacy_op_dir, legacy_state
        )
        state_path = cwd / ".triton-agent" / f"state-{op_dir.name}.json"

    # Initialize if needed.
    state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            state = load_json(state_path)
        except ValueError:
            state = {}

    if not state_path.is_file() or not (op_dir / "round_index.json").is_file():
        init_workspace(
            cwd, op_name, op_dir, mode, task_desc, task_json, gpu_kernel_ref, gpu_perf_csv,
            config, active_algorithm, run_tag,
        )
        state = load_json(state_path)

    # Ensure the state always reflects the active algorithm.
    state["algorithm"] = active_algorithm
    save_json(state_path, state)

    # Load the strategy for the selected algorithm. State-aware strategies (e.g. MCTS)
    # may need per-operator initialization and per-round result recording.
    strategy = load_strategy_for_algorithm(config, active_algorithm)
    round_index_for_init: list[dict[str, Any]] = []
    if (op_dir / "round_index.json").is_file():
        try:
            round_index_for_init = load_json(op_dir / "round_index.json")
        except ValueError:
            pass
    if hasattr(strategy, "init_for_operator"):
        strategy.init_for_operator(op_dir, config, round_index_for_init)

    current_timeout = timeout

    while True:
        state = load_json(state_path)
        current_round = state["current_round"]
        work_dir = Path(state["work_dir"])
        summary = load_json(work_dir / "summary.json")

        # baseline_dir is authoritative in round_index.json (written by transition).
        # summary.json is a per-round worker artifact and may be overwritten.
        baseline_dir_name = None
        round_index_path = op_dir / "round_index.json"
        if round_index_path.is_file():
            try:
                round_index = load_json(round_index_path)
                entry = next(
                    (r for r in round_index if r.get("round_index") == current_round), None
                )
                if entry:
                    baseline_dir_name = entry.get("baseline_dir")
            except (ValueError, TypeError):
                pass
        if not baseline_dir_name:
            baseline_dir_name = summary.get("baseline_dir")
        baseline_dir = op_dir / baseline_dir_name if baseline_dir_name else None

        global_baseline_dir = op_dir / "global_baseline"
        global_baseline_speedup = 1.0
        baseline_info_path = global_baseline_dir / "baseline_info.json"
        if baseline_info_path.is_file():
            try:
                baseline_info = load_json(baseline_info_path)
                gbs = baseline_info.get("baseline_speedup", 1.0)
                if isinstance(gbs, (int, float)) and gbs > 0:
                    global_baseline_speedup = float(gbs)
            except (ValueError, TypeError):
                pass

        experience_file = op_dir / "operator_experience.md"
        lineage_summary = build_lineage_summary(round_index, f"opt-round-{current_round}")

        # algorithm is immutable for this run; warn if summary.json disagrees.
        summary_algorithm = summary.get("algorithm")
        if summary_algorithm and summary_algorithm != active_algorithm:
            print(
                f"WARN: summary.json algorithm ({summary_algorithm}) differs from "
                f"active algorithm ({active_algorithm}); using active algorithm.",
                file=sys.stderr,
            )
        decision_metadata: dict[str, Any] = {}
        for key in ("mcts_parent_id", "mcts_selection_path", "mcts_pending_expansion"):
            if key in summary:
                decision_metadata[key] = summary[key]

        manifest_path = build_manifest(
            op_dir, op_name, current_round,
            baseline_dir,
            active_algorithm,
            summary.get("round_strategy", "exploration"),
            summary.get("analysis_policy", "pattern_entry"),
            summary.get("hypothesis", ""),
            summary.get("direction", ""),
            summary.get("evidence_sources", []),
            mode, task_desc, task_json, gpu_kernel_ref, gpu_perf_csv,
            config,
            arch=arch,
            lineage_summary=lineage_summary,
            experience_file=str(experience_file) if experience_file.is_file() else None,
            global_baseline_dir=global_baseline_dir,
            global_baseline_speedup=global_baseline_speedup,
            decision_metadata=decision_metadata,
        )

        dispatch_result = dispatch_round(manifest_path, current_timeout)
        sub_pid = dispatch_result.get("pid")
        if sub_pid:
            print(f"[orchestrator] Round {current_round} sub-agent PID: {sub_pid}")
            print(f"[orchestrator] To kill manually: kill {sub_pid}")

        if dispatch_result["status"] == "timeout":
            new_timeout = min(current_timeout * 2, max_worker_timeout)
            print(f"[orchestrator] Round {current_round} timed out after {current_timeout}s; "
                  f"increasing timeout to {new_timeout}s for next round")
            current_timeout = new_timeout
            state = load_json(state_path)
            state["performance_history"].append({
                "round": current_round,
                "speedup_vs_torch": 0.0,
                "target_reached": False,
                "timeout": True,
            })
            save_json(state_path, state)
            # Record the timeout/failure in state-aware strategies if possible.
            if hasattr(strategy, "record_round_result"):
                try:
                    strategy.record_round_result(
                        {"round_index": current_round, "status": "timeout", "best_speedup": 0.0},
                        work_dir,
                        context={"op_dir": str(op_dir), "current_round": current_round},
                    )
                except Exception as exc:
                    print(f"[orchestrator] Strategy record_round_result warning: {exc}")
        elif dispatch_result["status"] != "success":
            return {
                "status": "failed",
                "reason": f"dispatch_round failed: {dispatch_result.get('stderr', '')}",
                "round": current_round,
            }
        else:
            round_result = dispatch_result["round_result"]
            update_round_index(op_dir, round_result, active_algorithm)

            # Record the completed round in any state-aware strategy (e.g. MCTS)
            # before selecting the next parent/direction.
            if hasattr(strategy, "record_round_result"):
                try:
                    round_index = load_json(op_dir / "round_index.json")
                    current_entry = next(
                        (r for r in round_index if r.get("round_index") == current_round), None
                    )
                    if current_entry:
                        strategy.record_round_result(
                            current_entry,
                            work_dir,
                            context={"op_dir": str(op_dir), "current_round": current_round},
                        )
                except Exception as exc:
                    print(f"[orchestrator] Strategy record_round_result warning: {exc}")

            best_speedup = round_result.get("best_speedup", 0.0)
            round_index = load_json(op_dir / "round_index.json")
            non_idle_count = count_non_idle_rounds(round_index)

            # C1/C2 stop check.
            c1 = best_speedup >= target_speedup
            c2 = non_idle_count >= max_rounds

            # Ask the strategy for the next round decision.
            current_policy = summary.get("analysis_policy", "pattern_entry")
            try:
                decision = strategy.select_next_round(
                    round_index,
                    op_category=op_category,
                    current_policy=current_policy,
                    context={"op_dir": str(op_dir), "current_round": current_round},
                )
            except Exception as exc:
                return {
                    "status": "failed",
                    "reason": f"strategy.select_next_round failed: {exc}",
                    "round": current_round,
                }

            # If the strategy thinks we should stop, run the stop gate.
            if decision.algorithm_recommends_stop:
                stop_check = check_stop_decision(
                    op_dir, current_round, best_speedup, target_speedup, max_rounds,
                    algorithm_recommends_stop=True,
                )
                if stop_check.get("can_stop", False) or c1 or c2:
                    best_entry = max(
                        (r for r in round_index if r.get("status") == "success"),
                        key=lambda r: r.get("best_speedup", 0.0),
                        default=None,
                    )
                    final_report = {
                        "status": "stopped",
                        "reason": decision.stop_reason or ("C1" if c1 else "C2"),
                        "best_round": best_entry.get("round_index") if best_entry else None,
                        "best_speedup": best_entry.get("best_speedup", 0.0) if best_entry else 0.0,
                        "target_speedup": target_speedup,
                        "total_rounds": non_idle_count,
                    }
                    if best_entry:
                        src = op_dir / best_entry["round_dir"] / f"{op_name}_generated.py"
                        dst = op_dir / f"{op_name}_final.py"
                        shutil.copy2(src, dst)
                        final_report["final_code"] = str(dst)
                    _update_global_template(
                        cwd, op_name, op_category, best_entry, op_dir / "operator_experience.md"
                    )
                    return final_report

            if c1 or c2:
                stop_check = check_stop_decision(
                    op_dir, current_round, best_speedup, target_speedup, max_rounds
                )
                if stop_check.get("can_stop", False):
                    best_entry = max(
                        (r for r in round_index if r.get("status") == "success"),
                        key=lambda r: r.get("best_speedup", 0.0),
                        default=None,
                    )
                    final_report = {
                        "status": "stopped",
                        "reason": "C1" if c1 else "C2",
                        "best_round": best_entry.get("round_index") if best_entry else None,
                        "best_speedup": best_entry.get("best_speedup", 0.0) if best_entry else 0.0,
                        "target_speedup": target_speedup,
                        "total_rounds": non_idle_count,
                    }
                    if best_entry:
                        src = op_dir / best_entry["round_dir"] / f"{op_name}_generated.py"
                        dst = op_dir / f"{op_name}_final.py"
                        shutil.copy2(src, dst)
                        final_report["final_code"] = str(dst)
                    _update_global_template(
                        cwd, op_name, op_category, best_entry, op_dir / "operator_experience.md"
                    )
                    return final_report

            # Fill in any missing fields that the strategy did not provide.
            # Direction is now LLM-driven; select_next_direction returns
            # pattern: custom with an exploratory hypothesis.
            if decision.direction is None:
                if strategy.capabilities.can_select_round:
                    # Round-selection strategies (e.g., MCTS) intentionally leave
                    # direction to the sub-agent.
                    pass
                elif strategy.capabilities.can_propose_direction:
                    import direction_selector as ds
                    strategy_name, policy, direction, hypothesis, evidence, meta = (
                        ds.select_next_direction(
                            round_index,
                            op_category=op_category,
                            current_policy=current_policy,
                        )
                    )
                    decision.direction = direction
                    decision.hypothesis = hypothesis or decision.hypothesis
                    decision.round_strategy = strategy_name or decision.round_strategy
                    decision.analysis_policy = policy or decision.analysis_policy
                    decision.evidence_sources = evidence or decision.evidence_sources
                    if not decision.metadata:
                        decision.metadata = meta
                else:
                    decision.direction = "pattern: custom"

            if decision.baseline.mode == "none" and decision.direction:
                # Pick a baseline using the strategy's own scoring function.
                best_score = -1.0
                best_entry: dict[str, Any] | None = None
                for entry in round_index:
                    if entry.get("status") != "success":
                        continue
                    score = strategy.score_baseline_for_direction(
                        entry,
                        target_direction=decision.direction,
                        target_strategy=decision.round_strategy or "exploration",
                        context={"op_dir": str(op_dir)},
                    )
                    if score > best_score:
                        best_score = score
                        best_entry = entry
                if best_entry:
                    decision.baseline = BaselineProposal(
                        mode="single", round_dirs=[best_entry["round_dir"]]
                    )

            # Resolve the baseline file path. If the selected parent round has no
            # code (virtual MCTS node), fall back to the immutable global baseline.
            baseline_file = resolve_baseline(op_dir, decision, round_index, op_name)
            if baseline_file is not None:
                baseline_dir = baseline_file.parent
                # Ensure prepare_next_round receives a concrete baseline proposal.
                # Virtual parents or empty proposals map to the global baseline.
                proposed_dir = decision.baseline.round_dirs[0] if decision.baseline.round_dirs else ""
                concrete_file = (
                    op_dir / proposed_dir / f"{op_name}_generated.py"
                    if proposed_dir
                    else None
                )
                if (
                    decision.baseline.mode == "none"
                    or not proposed_dir
                    or concrete_file is None
                    or not concrete_file.is_file()
                ):
                    effective_dir = baseline_dir.relative_to(op_dir).as_posix()
                    decision.baseline = BaselineProposal(mode="single", round_dirs=[effective_dir])
            else:
                baseline_dir = None

            state = load_json(state_path)
            state["performance_history"].append({
                "round": current_round,
                "speedup_vs_torch": best_speedup,
                "target_reached": best_speedup >= target_speedup,
            })
            state["direction_history"] = build_direction_history(round_index)
            state["algorithm"] = active_algorithm
            save_json(state_path, state)

            # Prepare the next round directory.
            prepare_next_round(
                op_dir, current_round, decision, config, round_index, active_algorithm, op_category,
                global_baseline_dir="global_baseline",
                global_baseline_speedup=global_baseline_speedup,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton Optimizer Multi-Agent Orchestrator Loop")
    parser.add_argument("--op-name", required=True, help="Operator name")
    parser.add_argument("--algorithm", default=None,
                        help="Optimization algorithm (required when multiple algorithms are enabled). "
                             "Must match a discovered strategy name.")
    parser.add_argument("--run-tag", default="default",
                        help="Run tag to isolate multiple runs of the same operator/algorithm (default: default)")
    parser.add_argument("--mode", default="A", choices=["A", "B"], help="Input mode")
    parser.add_argument("--task-desc", required=True, help="Path to task description .py file")
    parser.add_argument("--task-json", default=None, help="Path to task JSON file (multi-case mode)")
    parser.add_argument("--gpu-kernel-ref", default=None, help="Path to GPU kernel reference file")
    parser.add_argument("--gpu-perf-csv", default=None, help="Path to GPU performance CSV")
    parser.add_argument("--timeout", type=int, default=7200, help="Initial sub CLI timeout in seconds")
    parser.add_argument("--op-category", default=None, help="Operator category (elementwise, matmul, normalization, ...)")
    parser.add_argument("--arch", default="ascend910b1", help="Target Ascend architecture (e.g. ascend910b3)")
    parser.add_argument(
        "--migrate-legacy",
        default=None,
        metavar="ALGORITHM",
        help="Migrate a legacy workspace (triton_ascend_output/{op_name}/) to the new naming and continue with ALGORITHM",
    )
    parser.add_argument(
        "--detect-legacy",
        action="store_true",
        help="Detect legacy workspace, print JSON report, and exit without modifying anything",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    result = run_orchestrator(
        cwd,
        args.op_name,
        args.mode,
        Path(args.task_desc).expanduser().resolve(),
        Path(args.task_json).expanduser().resolve() if args.task_json else None,
        Path(args.gpu_kernel_ref).expanduser().resolve() if args.gpu_kernel_ref else None,
        Path(args.gpu_perf_csv).expanduser().resolve() if args.gpu_perf_csv else None,
        args.timeout,
        op_category=args.op_category,
        arch=args.arch,
        algorithm=args.algorithm,
        run_tag=args.run_tag,
        migrate_legacy=args.migrate_legacy,
        detect_legacy=args.detect_legacy,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "stopped" else 1


if __name__ == "__main__":
    sys.exit(main())
