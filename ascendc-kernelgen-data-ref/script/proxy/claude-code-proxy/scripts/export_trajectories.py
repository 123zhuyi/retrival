#!/usr/bin/env python3
"""Export complete OpenAI-format trajectories for every session.

For each non-archived session we emit one JSON file:

    <out_dir>/<session_id>.json

Each file contains the session's main + subagent requests in chronological
order. Every step holds the exact OpenAI-style payload that was forwarded
to the backend (Qwen / other) plus the assistant reply, converted from the
Anthropic response_body back into the OpenAI assistant-message shape
(`content` + `tool_calls`). titler / count_tokens / external requests are
filtered out.

Usage:
    python scripts/export_trajectories.py [--db cc_traces/trace.db]
                                          [--out exports/trajectories]
                                          [--include-archived]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


KEPT_ROLES = ("main", "subagent")


def _json_load(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _strip_request_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop transport-only fields so the payload reads as a pure OpenAI body."""
    if not isinstance(payload, dict):
        return payload
    drop = {"api_key", "api_base", "extra_headers", "headers"}
    return {k: v for k, v in payload.items() if k not in drop}


def _anthropic_response_to_openai(response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert the stored Anthropic response into an OpenAI chat.completions-shaped reply."""
    if not isinstance(response, dict):
        return None

    blocks = response.get("content") or []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
        elif btype == "tool_use":
            raw_input = block.get("input")
            try:
                arguments = json.dumps(raw_input, ensure_ascii=False)
            except (TypeError, ValueError):
                arguments = json.dumps(str(raw_input), ensure_ascii=False)
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": arguments,
                    },
                }
            )

    message: Dict[str, Any] = {"role": "assistant"}
    message["content"] = "\n".join(text_parts) if text_parts else ""
    if tool_calls:
        message["tool_calls"] = tool_calls

    stop_reason = response.get("stop_reason")
    finish_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_map.get(stop_reason, stop_reason)

    return {
        "id": response.get("id"),
        "model": response.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": response.get("usage"),
    }


def export_session(conn: sqlite3.Connection, session_row: sqlite3.Row) -> Dict[str, Any]:
    session_id = session_row["session_id"]
    placeholders = ",".join("?" * len(KEPT_ROLES))
    request_rows = conn.execute(
        f"""
        SELECT trace_id, session_id, api, role_kind, agent_label,
               model_requested, model_mapped,
               started_at, completed_at, duration_ms,
               status, status_code, response_stop_reason,
               parent_trace_id, parent_agent_call_id, title,
               converted_request_json, response_body_json, error_json
        FROM requests
        WHERE session_id = ?
          AND role_kind IN ({placeholders})
          AND api = 'messages'
        ORDER BY started_ms ASC, trace_id ASC
        """,
        (session_id, *KEPT_ROLES),
    ).fetchall()

    steps: List[Dict[str, Any]] = []
    for r in request_rows:
        request_payload = _strip_request_payload(_json_load(r["converted_request_json"]))
        anthropic_response = _json_load(r["response_body_json"])
        openai_response = _anthropic_response_to_openai(anthropic_response)
        error = _json_load(r["error_json"])

        steps.append(
            {
                "trace_id": r["trace_id"],
                "role_kind": r["role_kind"],
                "agent_label": r["agent_label"],
                "model_requested": r["model_requested"],
                "model_mapped": r["model_mapped"],
                "started_at": r["started_at"],
                "completed_at": r["completed_at"],
                "duration_ms": r["duration_ms"],
                "status": r["status"],
                "status_code": r["status_code"],
                "stop_reason": r["response_stop_reason"],
                "parent_trace_id": r["parent_trace_id"],
                "parent_agent_call_id": r["parent_agent_call_id"],
                "title": r["title"],
                "request": request_payload,
                "response": openai_response,
                "error": error,
            }
        )

    agent_call_rows = conn.execute(
        """
        SELECT agent_call_id, parent_trace_id, tool_use_id, tool_name, agent_label,
               started_at, completed_at, duration_ms, status,
               input_preview, result_preview, result_trace_id,
               child_request_ids_json
        FROM agent_calls
        WHERE session_id = ?
        ORDER BY started_ms ASC, agent_call_id ASC
        """,
        (session_id,),
    ).fetchall()

    agent_calls: List[Dict[str, Any]] = []
    for ac in agent_call_rows:
        agent_calls.append(
            {
                "agent_call_id": ac["agent_call_id"],
                "parent_trace_id": ac["parent_trace_id"],
                "tool_use_id": ac["tool_use_id"],
                "tool_name": ac["tool_name"],
                "agent_label": ac["agent_label"],
                "started_at": ac["started_at"],
                "completed_at": ac["completed_at"],
                "duration_ms": ac["duration_ms"],
                "status": ac["status"],
                "input_preview": ac["input_preview"],
                "result_preview": ac["result_preview"],
                "result_trace_id": ac["result_trace_id"],
                "child_request_ids": _json_load(ac["child_request_ids_json"]) or [],
            }
        )

    return {
        "session_id": session_id,
        "origin": session_row["origin"],
        "first_seen": session_row["first_seen"],
        "last_seen": session_row["last_seen"],
        "cc_version": session_row["cc_version"],
        "cch": session_row["cch"],
        "step_count": len(steps),
        "title": next((s["title"] for s in steps if s.get("title")), None),
        "steps": steps,
        "agent_calls": agent_calls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="cc_traces/trace.db",
        help="Path to the trace.db SQLite file (default: cc_traces/trace.db).",
    )
    parser.add_argument(
        "--out",
        default="exports/trajectories",
        help="Output directory (default: exports/trajectories).",
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Also export sessions marked archived=1.",
    )
    parser.add_argument(
        "--session",
        action="append",
        help="Only export the given session_id(s). Can be passed multiple times.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: trace db not found: {db_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    session_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    has_archived = "archived" in session_cols

    where = []
    params: List[Any] = []
    if has_archived and not args.include_archived:
        where.append("archived = 0")
    if args.session:
        where.append(f"session_id IN ({','.join('?' * len(args.session))})")
        params.extend(args.session)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sessions = conn.execute(
        f"SELECT * FROM sessions {where_sql} ORDER BY last_seen DESC",
        params,
    ).fetchall()

    written = 0
    skipped_empty = 0
    for s in sessions:
        bundle = export_session(conn, s)
        if not bundle["steps"]:
            skipped_empty += 1
            continue
        safe_name = bundle["session_id"].replace("/", "_")
        target = out_dir / f"{safe_name}.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        written += 1
        print(f"  {target}  ({bundle['step_count']} steps)")

    print(
        f"done — wrote {written} session file(s) to {out_dir} "
        f"(skipped {skipped_empty} session(s) with no main/subagent requests)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
