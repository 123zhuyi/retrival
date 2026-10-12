#!/usr/bin/env python3
"""Shared workspace introspection utilities."""

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def find_op_name(op_dir: Path | str) -> str | None:
    """Return the authoritative operator name for a workspace.

    The workspace directory now includes the algorithm and run tag
    (e.g. ``layer_norm-mcts-default``), so ``op_dir.name`` is no longer
    the operator name. We look at stable metadata first, then generated
    code artifacts.
    """
    if isinstance(op_dir, str):
        op_dir = Path(op_dir)

    # 1. State file (most authoritative).
    state_dir = op_dir.parent.parent / ".triton-agent"
    state_path = state_dir / f"state-{op_dir.name}.json"
    if state_path.is_file():
        op_name = load_json(state_path).get("op_name")
        if isinstance(op_name, str) and op_name:
            return op_name

    # 2. Any task manifest in the operator directory.
    for manifest_path in sorted(op_dir.glob(".task_manifest_*.json")):
        op_name = load_json(manifest_path).get("op_name")
        if isinstance(op_name, str) and op_name:
            return op_name

    # 3. Immutable global baseline file.
    global_baseline = op_dir / "global_baseline"
    if global_baseline.is_dir():
        for path in sorted(global_baseline.glob("*_generated.py")):
            stem = path.stem
            if stem.endswith("_generated"):
                return stem[:-10]

    # 4. Round summaries (op_name field).
    for summary_path in sorted(op_dir.glob("opt-round-*/summary.json")):
        op_name = load_json(summary_path).get("op_name")
        if isinstance(op_name, str) and op_name:
            return op_name

    # 5. Generated code files in round directories.
    for rd in sorted(op_dir.glob("opt-round-*")):
        for path in sorted(rd.glob("*_generated.py")):
            stem = path.stem
            if stem.endswith("_generated"):
                return stem[:-10]

    return None
