#!/usr/bin/env python3
"""Continue MCTS optimization for chunk_gated_delta_fwd_h until 5 total rounds are completed."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/l30083907/cannbot-skills_5171_old/plugins-community/triton-optimizer-multi-agent")
SCRIPT_DIR = ROOT / "skills" / "triton-agent-loop" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from config_loader import read_config
from strategies import load_strategy_for_algorithm

from orchestrator_loop import (
    build_manifest,
    build_lineage_summary,
    dispatch_round,
    prepare_next_round,
    update_round_index,
    load_json,
    save_json,
)

OP_NAME = "chunk_gated_delta_fwd_h"
ALGORITHM = "mcts"
RUN_TAG = "mcts-0901"
OP_DIR = ROOT / "triton_ascend_output" / f"{OP_NAME}-{ALGORITHM}-{RUN_TAG}"
STATE_PATH = ROOT / ".triton-agent" / f"state-{OP_DIR.name}.json"
OP_CATEGORY = "chunked-recurrence"
MODE = "A"
TASK_DESC = ROOT.parent / "plugins-official" / "triton-op-generator" / f"{OP_NAME}_mc.py"
TASK_JSON = ROOT.parent / "plugins-official" / "triton-op-generator" / f"{OP_NAME}_mc.json"
GPU_KERNEL_REF = None
GPU_PERF_CSV = None
ARCH = "ascend910b1"

LOG_PATH = OP_DIR / "continue_mcts.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_completed_rounds(round_index: list) -> int:
    return sum(1 for r in round_index if r.get("status") == "success")


def prepare_or_get_current_round(
    strategy,
    config: dict,
    target_round: int,
) -> tuple[int, Path]:
    """Ensure the round directory for target_round exists; if not, prepare it."""
    state = load_json(STATE_PATH)
    current_round = state["current_round"]

    if current_round == target_round and (OP_DIR / f"opt-round-{target_round}" / "summary.json").is_file():
        return target_round, OP_DIR / f"opt-round-{target_round}"

    # Need to prepare target_round from current_round.
    round_index = load_json(OP_DIR / "round_index.json") if (OP_DIR / "round_index.json").is_file() else []
    decision = strategy.select_next_round(
        round_index,
        op_category=OP_CATEGORY,
        current_policy="pattern_entry",
        context={"op_dir": str(OP_DIR), "current_round": current_round},
    )
    if decision.algorithm_recommends_stop:
        raise RuntimeError(f"Strategy recommends stop: {decision.stop_reason}")

    log(
        f"Preparing Round {target_round} from Round {current_round}: "
        f"parent={decision.metadata.get('mcts_parent_id', 'root')}"
    )
    trans = prepare_next_round(
        OP_DIR,
        current_round,
        decision,
        config,
        round_index,
        ALGORITHM,
        op_category=OP_CATEGORY,
        global_baseline_dir="global_baseline",
        global_baseline_speedup=1.0,
    )
    log(f"transition_next_round: {json.dumps(trans, ensure_ascii=False)}")
    if trans.get("status") != "success":
        raise RuntimeError(f"Transition failed: {trans}")

    return trans["next_round"], OP_DIR / trans["next_round_dir"]


def run_round(
    strategy,
    config: dict,
    round_index: int,
    work_dir: Path,
) -> dict:
    """Build manifest, dispatch, and record result for a single round."""
    round_index_data = load_json(OP_DIR / "round_index.json") if (OP_DIR / "round_index.json").is_file() else []
    summary = load_json(work_dir / "summary.json")
    baseline_dir_name = summary.get("baseline_dir") or "global_baseline"
    baseline_dir = OP_DIR / baseline_dir_name

    lineage_summary = build_lineage_summary(round_index_data, work_dir.name)
    experience_file = OP_DIR / "operator_experience.md"
    global_baseline_dir = OP_DIR / "global_baseline"

    decision = strategy.select_next_round(
        round_index_data,
        op_category=OP_CATEGORY,
        current_policy=summary.get("analysis_policy", "pattern_entry"),
        context={"op_dir": str(OP_DIR), "current_round": round_index - 1},
    )

    manifest_path = build_manifest(
        OP_DIR,
        OP_NAME,
        round_index,
        baseline_dir,
        ALGORITHM,
        summary.get("round_strategy", "exploration"),
        summary.get("analysis_policy", "pattern_entry"),
        summary.get("hypothesis", ""),
        summary.get("direction", ""),
        summary.get("evidence_sources", ["mcts_tree"]),
        MODE,
        TASK_DESC,
        TASK_JSON,
        GPU_KERNEL_REF,
        GPU_PERF_CSV,
        config,
        arch=ARCH,
        lineage_summary=lineage_summary,
        experience_file=str(experience_file) if experience_file.is_file() else None,
        global_baseline_dir=global_baseline_dir,
        global_baseline_speedup=1.0,
        decision_metadata=summary.get("mcts_selection_path") and {
            "mcts_parent_id": summary.get("mcts_parent_id"),
            "mcts_selection_path": summary.get("mcts_selection_path"),
            "mcts_pending_expansion": summary.get("mcts_pending_expansion", True),
        } or decision.metadata,
    )
    log(f"Manifest: {manifest_path}")

    timeout = config.get("initial_worker_timeout", 7200)
    log(f"Dispatching Round {round_index} (timeout={timeout}s)")
    dispatch_result = dispatch_round(manifest_path, timeout)
    log(f"dispatch result: {json.dumps(dispatch_result, ensure_ascii=False, default=str)}")

    if dispatch_result.get("status") == "timeout":
        state = load_json(STATE_PATH)
        state["performance_history"].append({
            "round": round_index,
            "speedup_vs_torch": 0.0,
            "target_reached": False,
            "timeout": True,
        })
        save_json(STATE_PATH, state)
        return {"status": "timeout", "round_index": round_index}

    if dispatch_result.get("status") != "success":
        raise RuntimeError(f"Round {round_index} dispatch failed: {dispatch_result}")

    round_result = dispatch_result["round_result"]
    update_round_index(OP_DIR, round_result, ALGORITHM)

    # Record in MCTS tree.
    round_index_data = load_json(OP_DIR / "round_index.json")
    current_entry = next(
        (r for r in round_index_data if r.get("round_index") == round_index), None
    )
    if current_entry:
        try:
            strategy.record_round_result(
                current_entry,
                work_dir,
                context={"op_dir": str(OP_DIR), "current_round": round_index},
            )
            log(f"Recorded Round {round_index} in MCTS tree.")
        except Exception as exc:
            log(f"WARNING: record_round_result failed: {exc}")

    # Update state performance history.
    state = load_json(STATE_PATH)
    state["performance_history"].append({
        "round": round_index,
        "speedup_vs_torch": round_result.get("best_speedup", 0.0),
        "target_reached": bool(round_result.get("target_reached", False)),
    })
    save_json(STATE_PATH, state)

    return round_result


def main() -> int:
    log("=" * 60)
    log("Starting MCTS continuation: run until 5 total rounds completed")

    config = read_config(ROOT)
    strategy = load_strategy_for_algorithm(config, ALGORITHM)
    target_total = 5  # Round 1 already done; do 4 more.

    while True:
        round_index_data = load_json(OP_DIR / "round_index.json") if (OP_DIR / "round_index.json").is_file() else []
        completed = count_completed_rounds(round_index_data)
        log(f"Completed rounds so far: {completed}/{target_total}")
        if completed >= target_total:
            log("Target total rounds reached; stopping continuation.")
            break

        state = load_json(STATE_PATH)
        current_round = state["current_round"]
        work_dir = OP_DIR / f"opt-round-{current_round}"

        if not work_dir.is_dir() or not (work_dir / "summary.json").is_file():
            # Need to prepare this round.
            current_round, work_dir = prepare_or_get_current_round(strategy, config, current_round + 1)

        summary = load_json(work_dir / "summary.json")
        if summary.get("status") == "active":
            result = run_round(strategy, config, current_round, work_dir)
            log(
                f"Round {current_round} finished: status={result.get('status')}, "
                f"speedup={result.get('best_speedup', 0.0):.4f}"
            )
        else:
            log(f"Round {current_round} summary status={summary.get('status')}; preparing next round.")
            prepare_or_get_current_round(strategy, config, current_round + 1)

    log("MCTS continuation loop finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
