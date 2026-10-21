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
SessionEnd hook for triton-optimizer-multi-agent Agent Loop.

Cleans up .triton-agent/ runtime directory on session end.
- If any operator session is active (incomplete), preserves state for recovery and warns.
- If all sessions are completed/failed or no state exists, cleans up normally.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _is_worker_mode() -> bool:
    """Return True if running as a per-round sub-agent worker.

    Workers are ephemeral and must not clean up the orchestrator's
    .triton-agent/ runtime tree.
    """
    return os.environ.get("TRITON_OPTIMIZER_MODE", "").lower() == "worker"


def _resolve_workspace(payload: dict[str, object]) -> Path | None:
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd).expanduser().resolve()
    return None


def _load_all_states(runtime_dir: Path) -> list[dict[str, object]]:
    """Load all per-operator state files from .triton-agent/.

    Includes state-{op_name}-{algorithm}-{run_tag}.json files and the legacy
    state.json.
    """
    states: list[dict[str, object]] = []
    if not runtime_dir.is_dir():
        return states

    for path in sorted(runtime_dir.glob("state-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                states.append(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    legacy_path = runtime_dir / "state.json"
    if legacy_path.is_file():
        try:
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                states.append(data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    return states


def _cleanup_runtime_tree(runtime_dir: Path) -> None:
    if runtime_dir.name != ".triton-agent":
        return
    if runtime_dir.is_symlink() or runtime_dir.is_file():
        runtime_dir.unlink()
        return
    if runtime_dir.is_dir():
        shutil.rmtree(runtime_dir)


def main() -> int:
    # Workers are ephemeral; they do not own the orchestrator's runtime state.
    if _is_worker_mode():
        return 0

    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"triton-optimizer-multi-agent SessionEnd failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = _resolve_workspace(payload)
    if workspace is None:
        return 0

    runtime_dir = workspace / ".triton-agent"
    if not runtime_dir.is_dir():
        return 0

    # ── Clean up operator-level transition locks (if any) ──
    # 锁文件现在存储在 triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.transition_lock.json
    #（算子级），不再存储在全局 .triton-agent/ 下。清理所有算子级锁文件。
    ascend_output = workspace / "triton_ascend_output"
    if ascend_output.is_dir():
        for lock_file in ascend_output.glob("*/.transition_lock.json"):
            try:
                lock_file.unlink()
            except OSError:
                pass  # Non-fatal

    # Check if any operator session is active (incomplete)
    states = _load_all_states(runtime_dir)
    active_states = [s for s in states if s.get("status") == "active"]
    if active_states:
        names = [str(s.get("op_name", "unknown")) for s in active_states]
        print(
            f"triton-optimizer-multi-agent SessionEnd: Active session(s) for {', '.join(names)} "
            f"still active — preserving .triton-agent/ for recovery on next session start.",
            file=sys.stderr,
        )
        return 0  # Don't delete, preserve for recovery

    try:
        _cleanup_runtime_tree(runtime_dir)
    except Exception as exc:
        print(f"triton-optimizer-multi-agent SessionEnd failed open: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
