#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
SessionStart hook for triton-optimizer-multi-agent Agent Loop.

Detects workspace state from .triton-agent/state-{op_name}-{algorithm}-{run_tag}.json
files and injects context to inform the user about interrupted or active
optimization rounds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _is_worker_mode() -> bool:
    """Return True if running as a per-round sub-agent worker.

    Workers receive all required context via manifest.json and should not
    inject session-recovery context.
    """
    return os.environ.get("TRITON_OPTIMIZER_MODE", "").lower() == "worker"


def _resolve_workspace(payload: dict[str, object]) -> Path | None:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd).expanduser().resolve()
    return None


def _load_all_trackers(workspace: Path) -> list[dict[str, object]]:
    """Load all per-operator state files from .triton-agent/.

    Returns a list of active/completed/failed state dicts, one per
    state-{op_name}-{algorithm}-{run_tag}.json file. Legacy state.json is also
    loaded for backward compatibility.
    """
    state_dir = workspace / ".triton-agent"
    trackers: list[dict[str, object]] = []
    if not state_dir.is_dir():
        return trackers

    for path in sorted(state_dir.glob("state-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("current_round", 0) > 0:
                trackers.append(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # Backward compatibility: legacy state.json
    legacy_path = state_dir / "state.json"
    if legacy_path.is_file():
        try:
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("current_round", 0) > 0:
                trackers.append(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    return trackers


def _build_context_for_tracker(tracker: dict[str, object]) -> list[str]:
    status = tracker.get("status", "unknown")
    round_num = tracker.get("current_round", 0)
    work_dir = tracker.get("work_dir")
    op_name = tracker.get("op_name")
    algorithm = tracker.get("algorithm")
    run_tag = tracker.get("run_tag")
    last_phase = tracker.get("last_phase", 0)
    history = tracker.get("performance_history", [])

    lines: list[str] = []
    header = f"- **Operator `{op_name}`** — Round {round_num} (`{status}`)"
    if algorithm and run_tag:
        header += f" — algorithm `{algorithm}`, run_tag `{run_tag}`"
    if work_dir:
        header += f" — `{work_dir}`"
    lines.append(header)

    if isinstance(last_phase, int) and last_phase > 0:
        lines.append(f"  - Last phase: `Phase {last_phase}`")

    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict):
            sp = last.get("speedup_vs_torch")
            tr = last.get("target_reached", False)
            if sp is not None:
                lines.append(f"  - Last speedup: `{sp}`")
                lines.append(f"  - Target reached: `{tr}`")

    return lines


def _build_context(trackers: list[dict[str, object]]) -> str | None:
    if not trackers:
        return None

    if len(trackers) == 1:
        tracker = trackers[0]
        status = tracker.get("status", "unknown")
        round_num = tracker.get("current_round", 0)
        op_name = tracker.get("op_name", "unknown")
        algorithm = tracker.get("algorithm")
        run_tag = tracker.get("run_tag")
        lines = [
            "## Previous Optimization Round Detected",
            "",
            f"The workspace has an existing optimization session (Round {round_num}) "
            f"with status `{status}`.",
        ]
        if algorithm and run_tag:
            lines[-1] += f" Algorithm: `{algorithm}`, run_tag: `{run_tag}`."
        # Reuse per-tracker formatting but strip the leading bullet header
        per_tracker_lines = _build_context_for_tracker(tracker)
        for line in per_tracker_lines[1:]:
            if line.startswith("  - "):
                lines.append(line)
            else:
                lines.append(f"  {line}")

        if status == "active":
            lines.extend([
                "",
                "The last round appears to be incomplete. "
                "You can resume by asking the agent to **continue the previous round**, "
                "or start a fresh task to begin a new optimization session.",
            ])
        elif status == "completed":
            lines.extend([
                "",
                "The last round completed successfully. "
                "You can continue with a new round or start a fresh task.",
            ])
        elif status == "failed":
            lines.extend([
                "",
                "The last round failed. You may retry or start a new task.",
            ])
        return "\n".join(lines)

    # Multiple active/completed sessions
    lines = [
        "## Previous Optimization Sessions Detected",
        "",
        f"The workspace has {len(trackers)} existing optimization session(s):",
    ]
    for tracker in trackers:
        lines.extend(_build_context_for_tracker(tracker))

    active = [t for t in trackers if t.get("status") == "active"]
    if active:
        lines.extend([
            "",
            f"There are {len(active)} active session(s). You can resume a specific operator "
            "by asking the agent to **continue the previous round for `<op_name>`**,"
            "or start a fresh task for a new operator.",
        ])
    else:
        lines.extend([
            "",
            "All previous sessions are completed or failed. "
            "You can start a fresh task or retry a previous operator.",
        ])

    return "\n".join(lines)


def main() -> int:
    # Workers are spawned with a manifest; they do not need session recovery context.
    if _is_worker_mode():
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"triton-optimizer-multi-agent SessionStart failed open: {exc}", file=sys.stderr)
        return 0

    if not isinstance(payload, dict):
        return 0

    workspace = _resolve_workspace(payload)
    if workspace is None:
        return 0

    trackers = _load_all_trackers(workspace)
    context = _build_context(trackers)
    if context:
        print(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
