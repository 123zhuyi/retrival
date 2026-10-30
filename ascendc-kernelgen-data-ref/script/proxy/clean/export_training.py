#!/usr/bin/env python3
"""Export training-ready OpenAI-format trajectories with trial-and-error cleaning.

One session → one trajectory. We take each session's main-agent thread,
flatten it into a standard OpenAI chat-completions conversation (system /
user / assistant / tool), then run a second-pass cleaner that drops
"trial-and-error" steps so the surviving trajectory looks like a single
clean attempt (AAA + BBB + CCC → CCC).

Outputs (per session_id) under <out_dir>/:
    <session>.raw.json    flat trajectory before cleaning
    <session>.clean.json  after applying all enabled cleaning rules
    <session>.diff.json   which steps were dropped, by which rule, and why

Both .raw and .clean are usable as-is for SFT (each is a {messages, tools}
sample). .diff is for auditing the cleaner's decisions.

Cleaning rules (all on by default; disable with --no-<rule>):
    failed_retry       drop a failed tool step if a same-tool step with the
                       same key argument succeeds before the next user turn
    dedupe_read        drop Read(file=X) if X was already Read and no Write/
                       Edit happened in between
    compress_search    drop empty Grep/Glob steps within a run that ends in
                       a productive Grep/Glob
    overridden_write   drop Write(file=X) if X is overwritten by a later
                       Write before any Read consumed it

The DB is either the OpenAI/litellm trace (older runs) or a passthrough
Anthropic trace (66/trace.db) — we always work from the Anthropic source
(request_body_json + response_body_json) so both shapes are supported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


KEPT_ROLES = ("main",)  # subagent has its own session-internal flow; we model
                        # it as the main thread's opaque Agent/Task tool_call.

ALL_RULES = ("failed_retry", "dedupe_read", "compress_search", "overridden_write")

OPSNAME_PATTERNS = [
    re.compile(r"算子描述文件为\s*\S*?/(\d+_[A-Za-z][A-Za-z0-9_]*)\.py"),
    re.compile(r"输出到\s*\S*?/(\d+_[A-Za-z][A-Za-z0-9_]*)/?"),
    re.compile(r"/level\d+/(\d+_[A-Za-z][A-Za-z0-9_]*)\.py"),
]
_FNAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def extract_opsname(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Scan the first few user messages for the operator-name pattern used in
    the Ascend op-gen prompts. Returns sanitized opsname or None."""
    blob = []
    for m in messages[:8]:
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, str):
                blob.append(content)
        if sum(len(s) for s in blob) > 8000:
            break
    text = "\n".join(blob)
    for pat in OPSNAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = m.group(1)
            return _FNAME_SAFE_RE.sub("_", name).strip("._-") or None
    return None


ERROR_TEXT_RE = re.compile(
    r"(?:^|\b)(error[:!]|exception|traceback|fatal:|command not found|"
    r"no such file|permission denied|exit code\s*[1-9]|syntax error|"
    r"<tool_use_error|errno\s*[1-9])",
    re.I,
)
EMPTY_HINTS = (
    "no files found", "no matches found", "0 results", "no results",
    "no such files", "no matching", "did not match any files",
)


# ---------------------------------------------------------------------------
# Anthropic → OpenAI conversion
# ---------------------------------------------------------------------------

def _system_to_text(sys_field: Any) -> str:
    if sys_field is None:
        return ""
    if isinstance(sys_field, str):
        return sys_field
    if isinstance(sys_field, list):
        parts = []
        for blk in sys_field:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                parts.append(blk["text"])
            elif isinstance(blk, str):
                parts.append(blk)
        return "\n".join(parts)
    return str(sys_field)


def _content_to_text(content: Any) -> str:
    """Flatten Anthropic content (str or list of blocks) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
                elif t == "image":
                    parts.append("[image]")
                elif t == "thinking" and isinstance(blk.get("thinking"), str):
                    parts.append(blk["thinking"])
                else:
                    # tool_use / tool_result handled separately; ignore here
                    pass
        return "\n".join(p for p in parts if p)
    return str(content)


def _tool_result_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    parts.append(blk["text"])
                elif t == "image":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(blk, ensure_ascii=False))
            else:
                parts.append(str(blk))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _tool_use_to_openai_call(blk: Dict[str, Any]) -> Dict[str, Any]:
    raw_input = blk.get("input")
    try:
        arguments = json.dumps(raw_input or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        arguments = json.dumps({"_raw": str(raw_input)}, ensure_ascii=False)
    return {
        "id": blk.get("id"),
        "type": "function",
        "function": {
            "name": blk.get("name"),
            "arguments": arguments,
        },
    }


def anthropic_message_to_openai(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert one Anthropic message into 0+ OpenAI-format messages.

    User messages containing tool_result blocks expand into one `role:"tool"`
    message per tool_result, plus an optional user message for any text.
    Assistant messages with tool_use blocks become a single assistant message
    with `tool_calls`, with text + thinking concatenated into `content`.
    """
    role = msg.get("role")
    content = msg.get("content")

    # User message: split into tool messages + optional user text.
    if role == "user":
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if not isinstance(content, list):
            return [{"role": "user", "content": _content_to_text(content)}]

        out: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for blk in content:
            if not isinstance(blk, dict):
                if isinstance(blk, str):
                    text_parts.append(blk)
                continue
            t = blk.get("type")
            if t == "tool_result":
                text = _tool_result_content_to_text(blk.get("content"))
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": blk.get("tool_use_id"),
                    "content": text,
                }
                if blk.get("is_error"):
                    tool_msg["_is_error"] = True  # stripped before final emit
                out.append(tool_msg)
            elif t == "text" and isinstance(blk.get("text"), str):
                text_parts.append(blk["text"])
            elif t == "image":
                text_parts.append("[image]")
        text_joined = "\n".join(p for p in text_parts if p).strip()
        if text_joined:
            out.append({"role": "user", "content": text_joined})
        return out

    # Assistant message: collapse thinking+text into content, tool_use → tool_calls.
    if role == "assistant":
        thinking_parts: List[str] = []
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for blk in content:
                if not isinstance(blk, dict):
                    if isinstance(blk, str):
                        text_parts.append(blk)
                    continue
                t = blk.get("type")
                if t == "text" and isinstance(blk.get("text"), str):
                    text_parts.append(blk["text"])
                elif t == "thinking" and isinstance(blk.get("thinking"), str):
                    thinking_parts.append(blk["thinking"])
                elif t == "tool_use":
                    tool_calls.append(_tool_use_to_openai_call(blk))

        parts: List[str] = []
        if thinking_parts:
            parts.append("<think>\n" + "\n".join(thinking_parts).strip() + "\n</think>")
        if text_parts:
            parts.append("\n".join(text_parts).strip())
        content_str = "\n\n".join(p for p in parts if p) or ""

        out_msg: Dict[str, Any] = {"role": "assistant", "content": content_str}
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        return [out_msg]

    # Anything else: best-effort
    return [{"role": role or "user", "content": _content_to_text(content)}]


def anthropic_response_to_openai_message(response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(response, dict):
        return None
    msg = {"role": "assistant", "content": response.get("content")}
    converted = anthropic_message_to_openai(msg)
    return converted[0] if converted else None


def anthropic_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(tools, list):
        return out
    for t in tools:
        if not isinstance(t, dict):
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description"),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


# ---------------------------------------------------------------------------
# Build the raw flat trajectory for a session
# ---------------------------------------------------------------------------

def _get_anthropic_body(row: sqlite3.Row) -> Dict[str, Any]:
    """Return the original Anthropic request body regardless of trace mode."""
    if row["request_body_json"]:
        try:
            return json.loads(row["request_body_json"])
        except json.JSONDecodeError:
            pass
    if row["converted_request_json"]:
        try:
            conv = json.loads(row["converted_request_json"])
        except json.JSONDecodeError:
            conv = {}
        if isinstance(conv, dict) and isinstance(conv.get("body"), dict):
            return conv["body"]
    return {}


COMPACTION_MARKER = "[--- context compaction boundary (Claude Code /compact) ---]"


def _segment_main_requests(rows: List[sqlite3.Row]) -> List[List[int]]:
    """Split chronologically-ordered main requests into segments at compaction
    points. A compaction is signaled by message_count dropping vs the prior
    request. Within a segment, message_count grows monotonically; we keep only
    the last request of each segment (the tip of that segment's chain)."""
    segments: List[List[int]] = []
    current: List[int] = []
    prev_count = -1
    for i, r in enumerate(rows):
        mc = r["message_count"] or 0
        if current and mc < prev_count:
            segments.append(current)
            current = []
        current.append(i)
        prev_count = mc
    if current:
        segments.append(current)
    return segments


_TRAJECTORY_SELECT = """
    SELECT trace_id, message_count, started_ms,
           request_body_json, converted_request_json, response_body_json,
           model_requested, model_mapped, response_stop_reason
"""


def _build_trajectory_from_rows(
    rows: List[sqlite3.Row],
    session_row: sqlite3.Row,
    kind_label: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble one flat OpenAI trajectory from a chronologically ordered list
    of request rows. Detects compaction (message_count drops) and stitches
    each segment together with a boundary marker. Returns None if there is
    no usable content."""
    if not rows:
        return None

    segments = _segment_main_requests(rows)
    segment_tips = [rows[seg[-1]] for seg in segments]

    messages: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []
    segment_summaries: List[Dict[str, Any]] = []

    for seg_idx, (seg_indices, tip) in enumerate(zip(segments, segment_tips)):
        body = _get_anthropic_body(tip)
        if not body:
            continue

        # v2.1.220+ normalization: system messages may be in the messages array
        # (OpenAI style) rather than in the top-level system field.  Extract them
        # so they don't appear mid-conversation and trigger R4 violations.
        raw_messages = body.get("messages")
        if raw_messages:
            sys_blocks = []
            non_sys = []
            for m in raw_messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    content = m.get("content", "")
                    if isinstance(content, str):
                        sys_blocks.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        sys_blocks.extend(content)
                else:
                    non_sys.append(m)
            if sys_blocks:
                existing = body.get("system")
                if existing is None:
                    body["system"] = sys_blocks
                elif isinstance(existing, str):
                    body["system"] = [{"type": "text", "text": existing}] + sys_blocks
                elif isinstance(existing, list):
                    body["system"] = list(existing) + sys_blocks
                body["messages"] = non_sys

        seg_message_start = len(messages)

        if seg_idx == 0:
            system_text = _system_to_text(body.get("system"))
            if system_text:
                messages.append({"role": "system", "content": system_text})
        else:
            messages.append({
                "role": "system",
                "content": COMPACTION_MARKER,
                "_compaction_marker": True,
            })

        for am in body.get("messages") or []:
            if isinstance(am, dict):
                messages.extend(anthropic_message_to_openai(am))

        response_body = None
        if tip["response_body_json"]:
            try:
                response_body = json.loads(tip["response_body_json"])
            except json.JSONDecodeError:
                response_body = None
        final_assistant = anthropic_response_to_openai_message(response_body)
        if final_assistant:
            messages.append(final_assistant)

        tools = anthropic_tools_to_openai(body.get("tools")) or tools

        segment_summaries.append({
            "segment_index": seg_idx,
            "tip_trace_id": tip["trace_id"],
            "tip_message_count": tip["message_count"],
            "tip_stop_reason": tip["response_stop_reason"],
            "n_requests": len(seg_indices),
            "first_started_ms": rows[seg_indices[0]]["started_ms"],
            "last_started_ms": rows[seg_indices[-1]]["started_ms"],
            "openai_message_range": [seg_message_start, len(messages)],
        })

    meta: Dict[str, Any] = {
        "trajectory_kind": kind_label,
        "n_segments": len(segments),
        "n_compactions": max(0, len(segments) - 1),
        "n_requests": len(rows),
        "tip_trace_ids": [t["trace_id"] for t in segment_tips],
        "segments": segment_summaries,
        "model_requested": segment_tips[-1]["model_requested"] if segment_tips else None,
        "model_mapped": segment_tips[-1]["model_mapped"] if segment_tips else None,
        "final_stop_reason": segment_tips[-1]["response_stop_reason"] if segment_tips else None,
        "first_seen": session_row["first_seen"],
        "last_seen": session_row["last_seen"],
    }
    if extra_meta:
        meta.update(extra_meta)

    return {
        "session_id": session_row["session_id"],
        "tools": tools,
        "messages": messages,
        "_meta": meta,
    }


def build_raw_trajectory(conn: sqlite3.Connection, session_row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    sid = session_row["session_id"]
    rows = conn.execute(
        f"""
        {_TRAJECTORY_SELECT}
        FROM requests
        WHERE session_id = ? AND role_kind = 'main' AND api = 'messages'
        ORDER BY started_ms ASC
        """,
        (sid,),
    ).fetchall()

    # v2.1.220+ fallback: all requests are role_kind='main' (including subagent ones).
    # Exclude subagent time windows so the main trajectory only contains parent-session
    # requests.  Subagent windows are inferred from agent_calls.parent_trace_id.
    if rows:
        subagent_exists = conn.execute(
            "SELECT COUNT(*) FROM requests "
            "WHERE session_id = ? AND role_kind = 'subagent' AND api = 'messages'",
            (sid,),
        ).fetchone()[0]
        if not subagent_exists:
            acs = conn.execute(
                "SELECT parent_trace_id FROM agent_calls "
                "WHERE session_id = ? AND tool_name IN ('Agent', 'Task')",
                (sid,),
            ).fetchall()
            if acs:
                spawn_starts: List[int] = []
                for ac in acs:
                    parent_tid = ac["parent_trace_id"]
                    if parent_tid:
                        spawn_row = conn.execute(
                            "SELECT started_ms FROM requests WHERE trace_id = ?",
                            (parent_tid,),
                        ).fetchone()
                        if spawn_row and spawn_row[0] is not None:
                            spawn_starts.append(spawn_row[0])
                if spawn_starts:
                    spawn_starts.sort()
                    # Build exclusion windows: [spawn_i, spawn_{i+1}) for i<n, [spawn_n, ∞)
                    exclude_ranges: List[Tuple[int, float]] = []
                    for i, sp in enumerate(spawn_starts):
                        if i + 1 < len(spawn_starts):
                            exclude_ranges.append((sp, spawn_starts[i + 1]))
                        else:
                            exclude_ranges.append((sp, float("inf")))
                    rows = [r for r in rows
                            if r["started_ms"] is not None
                            and not any(lo <= r["started_ms"] < hi for lo, hi in exclude_ranges)]

    return _build_trajectory_from_rows(list(rows), session_row, "main")


def _synthesize_agent_calls_from_db_tool_events(
    conn: sqlite3.Connection, session_id: str
) -> List[Dict[str, Any]]:
    """DB version of _synthesize_agent_calls_from_tool_events.

    For old trace.db files where the agent_calls table is empty but tool_events
    is populated, rebuild agent_call rows from Agent/Task tool_use_history events.
    This mirrors the JSON-path fallback so that old DB exports get subagent
    trajectories too.
    """
    # Check whether tool_events table exists (very old DBs may not have it)
    cols = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "tool_events" not in cols:
        return []

    sub_starts = [
        r["started_ms"] for r in
        conn.execute(
            "SELECT started_ms FROM requests "
            "WHERE session_id = ? AND role_kind = 'subagent' AND api = 'messages' "
            "ORDER BY started_ms ASC",
            (session_id,),
        ).fetchall()
        if r["started_ms"] is not None
    ]

    agent_events = conn.execute(
        """
        SELECT trace_id, tool_use_id, tool_name, agent_label,
               event_time, event_time_ms, input_preview
        FROM tool_events
        WHERE session_id = ? AND event_type = 'tool_use_history'
          AND tool_name IN ('Agent', 'Task')
          AND tool_use_id IS NOT NULL
        ORDER BY event_time_ms ASC
        """,
        (session_id,),
    ).fetchall()

    if not agent_events:
        return []

    out = []
    for idx, te in enumerate(agent_events):
        tu_id = te["tool_use_id"]
        seed = f"{te['trace_id'] or ''}|{tu_id}"
        agent_call_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:20]
        if len(agent_events) == 1:
            inferred_start = sub_starts[0] if sub_starts else te["event_time_ms"]
        else:
            block_size = max(1, len(sub_starts) // len(agent_events))
            block_idx = idx * block_size
            inferred_start = (sub_starts[block_idx] if block_idx < len(sub_starts)
                              else te["event_time_ms"])
        out.append({
            "agent_call_id": agent_call_id,
            "agent_label": te["agent_label"],
            "tool_name": te["tool_name"],
            "started_ms": inferred_start,
            "status": "unknown",
            "input_preview": te["input_preview"],
        })
    return out


def build_subagent_trajectories(
    conn: sqlite3.Connection, session_row: sqlite3.Row
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return [(agent_call_id, subagent_type, bundle), ...] for every Agent/Task
    invocation in this session.

    Handles three historical data shapes:
      new-style  — all child requests carry parent_agent_call_id (current trace code)
      old-style  — only the first child request carries parent_agent_call_id; the rest
                   chain via parent_trace_id but the 1→2 link is missing (tracing bug)
      legacy-export — agent_calls table absent; all requests share one parent_agent_call_id

    Partitioning strategy: use each agent_call's "seed start time" (min started_ms of
    directly-linked requests) as a time boundary.  All subagent requests from that time
    up to the next seed time belong to this agent_call.  This survives broken
    parent_trace_id chains because it doesn't depend on link completeness.
    """
    sid = session_row["session_id"]
    acs = conn.execute(
        """
        SELECT agent_call_id, agent_label, tool_name, started_ms, status, input_preview, parent_trace_id
        FROM agent_calls
        WHERE session_id = ? AND tool_name IN ('Agent', 'Task')
        ORDER BY started_ms ASC
        """,
        (sid,),
    ).fetchall()

    # Fallback: agent_calls table absent — synthesize from distinct parent_agent_call_id
    if not acs:
        synthetic = conn.execute(
            """
            SELECT parent_agent_call_id AS agent_call_id, agent_label,
                   MIN(started_ms) AS started_ms
            FROM requests
            WHERE session_id = ? AND role_kind = 'subagent'
              AND parent_agent_call_id IS NOT NULL
            GROUP BY parent_agent_call_id
            ORDER BY started_ms ASC
            """,
            (sid,),
        ).fetchall()
        if not synthetic:
            # Third fallback: synthesize from tool_events (old DBs that predate
            # both agent_calls table AND parent_agent_call_id on subagent requests)
            acs = _synthesize_agent_calls_from_db_tool_events(conn, sid)
        else:
            acs = [{"agent_call_id": r["agent_call_id"], "agent_label": r["agent_label"],
                    "tool_name": "Agent", "started_ms": r["started_ms"],
                    "status": "unknown", "input_preview": None}
                   for r in synthetic]

    # Seed start time for each agent_call = min started_ms of directly-linked requests.
    # If no requests carry parent_agent_call_id (very old dumps), fall back to the
    # agent_call's own started_ms — the earliest subagent request is the next thing
    # that fires after the Agent tool_use, so this is still a valid lower bound.
    seed_starts: List[Optional[int]] = []
    for ac in acs:
        row = conn.execute(
            "SELECT MIN(started_ms) FROM requests "
            "WHERE session_id = ? AND parent_agent_call_id = ? AND api = 'messages'",
            (sid, ac["agent_call_id"]),
        ).fetchone()
        t = row[0] if row else None
        if t is None:
            t = ac["started_ms"]
        seed_starts.append(t)

    # All subagent requests for this session, sorted chronologically
    all_rows = conn.execute(
        f"""
        {_TRAJECTORY_SELECT}
        FROM requests
        WHERE session_id = ? AND role_kind = 'subagent' AND api = 'messages'
        ORDER BY started_ms ASC
        """,
        (sid,),
    ).fetchall()

    # v2.1.220+ fallback: subagent requests are not tagged with role_kind='subagent';
    # they all carry role_kind='main'. Use agent_call.parent_trace_id to find the
    # spawn-point request; all requests after it belong to the subagent.
    if not all_rows and acs:
        spawn_starts: List[Optional[int]] = []
        for ac in acs:
            parent_tid = ac["parent_trace_id"]
            if parent_tid:
                spawn_row = conn.execute(
                    "SELECT started_ms FROM requests WHERE trace_id = ?",
                    (parent_tid,),
                ).fetchone()
                if spawn_row and spawn_row[0] is not None:
                    spawn_starts.append(spawn_row[0])
                else:
                    spawn_starts.append(ac["started_ms"])
            else:
                spawn_starts.append(ac["started_ms"])

        all_rows = conn.execute(
            f"""
            {_TRAJECTORY_SELECT}
            FROM requests
            WHERE session_id = ? AND role_kind = 'main' AND api = 'messages'
            ORDER BY started_ms ASC
            """,
            (sid,),
        ).fetchall()

        seed_starts = spawn_starts

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for i, ac in enumerate(acs):
        t_start = seed_starts[i]
        if t_start is None:
            continue
        t_end = next((seed_starts[j] for j in range(i + 1, len(acs))
                      if seed_starts[j] is not None), None)
        rows = [r for r in all_rows
                if r["started_ms"] is not None
                and r["started_ms"] >= t_start
                and (t_end is None or r["started_ms"] < t_end)]
        if not rows:
            continue
        bundle = _build_trajectory_from_rows(
            rows,
            session_row,
            "subagent",
            extra_meta={
                "agent_call_id": ac["agent_call_id"],
                "subagent_type": ac["agent_label"],
                "agent_call_status": ac["status"],
                "agent_call_input_preview": ac["input_preview"],
            },
        )
        if bundle and len(bundle["messages"]) >= 2:
            out.append((ac["agent_call_id"], ac["agent_label"] or "subagent", bundle))
    return out


# ---------------------------------------------------------------------------
# Step grouping
# ---------------------------------------------------------------------------

def group_steps(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group flat messages into logical steps for the cleaner."""
    steps: List[Dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        role = m.get("role")
        if role in ("system", "user"):
            steps.append({"kind": role, "message_indices": [i]})
            i += 1
        elif role == "assistant":
            tool_calls = m.get("tool_calls") or []
            step = {
                "kind": "assistant_tool" if tool_calls else "assistant_text",
                "message_indices": [i],
                "tool_calls": list(tool_calls),
                "tool_result_indices": [],
            }
            i += 1
            if tool_calls:
                expected_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
                while i < n and messages[i].get("role") == "tool":
                    if messages[i].get("tool_call_id") in expected_ids:
                        step["tool_result_indices"].append(i)
                        step["message_indices"].append(i)
                        i += 1
                    else:
                        break
            steps.append(step)
        elif role == "tool":
            steps.append({"kind": "orphan_tool", "message_indices": [i]})
            i += 1
        else:
            steps.append({"kind": "other", "message_indices": [i]})
            i += 1
    return steps


def tool_result_for(messages: List[Dict[str, Any]], step: Dict[str, Any], tool_call_id: str) -> Optional[Dict[str, Any]]:
    for idx in step.get("tool_result_indices", []):
        m = messages[idx]
        if m.get("tool_call_id") == tool_call_id:
            return m
    return None


# ---------------------------------------------------------------------------
# Status / key extraction
# ---------------------------------------------------------------------------

def is_error_content(text: str, is_error_flag: bool = False) -> bool:
    if is_error_flag:
        return True
    if not text:
        return False
    return bool(ERROR_TEXT_RE.search(text))


def is_empty_content(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    low = s.lower()
    if any(h in low for h in EMPTY_HINTS):
        return True
    if s in ("[]", "{}", "()"):
        return True
    return False


def classify_step_status(messages: List[Dict[str, Any]], step: Dict[str, Any]) -> str:
    if step["kind"] != "assistant_tool":
        return "n/a"
    tool_results = [messages[idx] for idx in step["tool_result_indices"]]
    if not tool_results:
        return "no_result"
    n_fail = 0
    n_empty = 0
    n_total = len(tool_results)
    for tr in tool_results:
        content = tr.get("content") or ""
        is_err = bool(tr.get("_is_error"))
        if is_error_content(content, is_err):
            n_fail += 1
        elif is_empty_content(content):
            n_empty += 1
    if n_fail == n_total:
        return "failure"
    if n_empty == n_total:
        return "empty"
    if n_fail + n_empty == n_total and n_fail > 0:
        return "failure"  # mix of failed + empty → call it failure
    if n_fail > 0 or n_empty > 0:
        return "mixed"
    return "success"


def _parse_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def tool_call_name(tool_call) -> str:
    if not isinstance(tool_call, dict):
        return ""
    return (tool_call.get("function") or {}).get("name") or ""


def tool_call_key(tool_call) -> Tuple[Any, ...]:
    """Stable 'same intent' key for grouping retry attempts."""
    if not isinstance(tool_call, dict):
        return ("",)
    name = tool_call_name(tool_call)
    args = _parse_args(tool_call)
    if name == "Bash":
        cmd = " ".join(str(args.get("command", "")).split())[:120]
        return (name, cmd)
    if name in ("Read", "Write", "Edit", "MultiEdit", "NotebookEdit"):
        return (name, args.get("file_path", ""))
    if name in ("Grep", "Glob"):
        return (name, args.get("pattern", ""), args.get("path", ""))
    if name in ("Agent", "Task"):
        return (name, args.get("subagent_type") or args.get("agent_type") or "",
                str(args.get("description", ""))[:60])
    if name == "WebFetch":
        return (name, args.get("url", ""))
    try:
        return (name, json.dumps(args, sort_keys=True, ensure_ascii=False)[:160])
    except (TypeError, ValueError):
        return (name, str(args)[:160])


def step_primary_call(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    calls = step.get("tool_calls") or []
    return calls[0] if calls else None


# ---------------------------------------------------------------------------
# Cleaning passes
# ---------------------------------------------------------------------------

def _find_next_user(steps: List[Dict[str, Any]], start: int) -> int:
    for j in range(start, len(steps)):
        if steps[j]["kind"] == "user":
            return j
    return len(steps)


def rule_failed_retry(messages: List[Dict[str, Any]], steps: List[Dict[str, Any]],
                      deletions: List[Dict[str, Any]]) -> int:
    """Drop a single-tool_call failed step if a later step within the same user
    segment retries the same (tool, key) and succeeds."""
    n_dropped = 0
    for i, step in enumerate(steps):
        if step.get("_deleted") or step["kind"] != "assistant_tool":
            continue
        if len(step.get("tool_calls") or []) != 1:
            continue
        status = classify_step_status(messages, step)
        if status not in ("failure", "empty"):
            continue
        key = tool_call_key(step["tool_calls"][0])
        end = _find_next_user(steps, i + 1)
        for j in range(i + 1, end):
            cand = steps[j]
            if cand.get("_deleted") or cand["kind"] != "assistant_tool":
                continue
            if len(cand.get("tool_calls") or []) != 1:
                continue
            if tool_call_key(cand["tool_calls"][0]) != key:
                continue
            if classify_step_status(messages, cand) == "success":
                step["_deleted"] = True
                n_dropped += 1
                deletions.append({
                    "rule": "failed_retry",
                    "step_index": i,
                    "tool": key[0],
                    "key": list(key[1:]),
                    "failed_status": status,
                    "retried_step_index": j,
                    "dropped_message_indices": step["message_indices"],
                })
                break
    return n_dropped


def rule_dedupe_read(messages: List[Dict[str, Any]], steps: List[Dict[str, Any]],
                     deletions: List[Dict[str, Any]]) -> int:
    n_dropped = 0
    last_read_at: Dict[str, int] = {}
    for i, step in enumerate(steps):
        if step.get("_deleted") or step["kind"] != "assistant_tool":
            continue
        if len(step.get("tool_calls") or []) != 1:
            # multi-tool: cautiously invalidate any file these touch
            for tc in step.get("tool_calls") or []:
                args = _parse_args(tc)
                fp = args.get("file_path")
                if fp and tool_call_name(tc) in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
                    last_read_at.pop(fp, None)
            continue
        tc = step["tool_calls"][0]
        name = tool_call_name(tc)
        args = _parse_args(tc)
        if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            fp = args.get("file_path")
            if fp:
                last_read_at.pop(fp, None)
            continue
        if name == "Bash":
            # might mutate any file — be conservative, invalidate all
            last_read_at.clear()
            continue
        if name != "Read":
            continue
        fp = args.get("file_path")
        if not fp:
            continue
        if classify_step_status(messages, step) != "success":
            continue
        if fp in last_read_at:
            step["_deleted"] = True
            n_dropped += 1
            deletions.append({
                "rule": "dedupe_read",
                "step_index": i,
                "file_path": fp,
                "first_read_step_index": last_read_at[fp],
                "dropped_message_indices": step["message_indices"],
            })
        else:
            last_read_at[fp] = i
    return n_dropped


def rule_compress_search(messages: List[Dict[str, Any]], steps: List[Dict[str, Any]],
                         deletions: List[Dict[str, Any]]) -> int:
    """Within a run of consecutive Grep/Glob steps, keep only the productive one(s)."""
    n_dropped = 0
    i = 0
    while i < len(steps):
        s = steps[i]
        if s.get("_deleted") or s["kind"] != "assistant_tool":
            i += 1
            continue
        if len(s.get("tool_calls") or []) != 1:
            i += 1
            continue
        if tool_call_name(s["tool_calls"][0]) not in ("Grep", "Glob"):
            i += 1
            continue
        # collect consecutive Grep/Glob run (skipping already-deleted)
        run = []
        j = i
        while j < len(steps):
            sj = steps[j]
            if sj.get("_deleted"):
                j += 1
                continue
            if sj["kind"] != "assistant_tool" or len(sj.get("tool_calls") or []) != 1:
                break
            if tool_call_name(sj["tool_calls"][0]) not in ("Grep", "Glob"):
                break
            run.append(j)
            j += 1
        if len(run) >= 2:
            # if at least one is productive, drop the empty/failed ones
            productive = [k for k in run if classify_step_status(messages, steps[k]) == "success"]
            if productive:
                for k in run:
                    if k in productive:
                        continue
                    st = classify_step_status(messages, steps[k])
                    if st in ("empty", "failure"):
                        steps[k]["_deleted"] = True
                        n_dropped += 1
                        deletions.append({
                            "rule": "compress_search",
                            "step_index": k,
                            "kept_step_indices": productive,
                            "dropped_status": st,
                            "dropped_message_indices": steps[k]["message_indices"],
                        })
        i = j
    return n_dropped


def rule_overridden_write(messages: List[Dict[str, Any]], steps: List[Dict[str, Any]],
                          deletions: List[Dict[str, Any]]) -> int:
    n_dropped = 0
    last_write_at: Dict[str, int] = {}
    for i, step in enumerate(steps):
        if step.get("_deleted") or step["kind"] != "assistant_tool":
            continue
        if len(step.get("tool_calls") or []) != 1:
            # multi-tool: invalidate everything (conservative)
            last_write_at.clear()
            continue
        tc = step["tool_calls"][0]
        name = tool_call_name(tc)
        args = _parse_args(tc)
        fp = args.get("file_path")
        if name == "Read" and fp:
            last_write_at.pop(fp, None)
            continue
        if name == "Bash":
            last_write_at.clear()
            continue
        if name == "Write" and fp:
            if classify_step_status(messages, step) != "success":
                continue
            if fp in last_write_at:
                prev = last_write_at[fp]
                if not steps[prev].get("_deleted"):
                    steps[prev]["_deleted"] = True
                    n_dropped += 1
                    deletions.append({
                        "rule": "overridden_write",
                        "step_index": prev,
                        "file_path": fp,
                        "overridden_by_step_index": i,
                        "dropped_message_indices": steps[prev]["message_indices"],
                    })
            last_write_at[fp] = i
            continue
        if name in ("Edit", "MultiEdit", "NotebookEdit") and fp:
            # Edits are incremental and hard to safely override-merge; don't drop
            # successful intermediates. They will only be removed via failed_retry.
            last_write_at.pop(fp, None)
    return n_dropped


# ---------------------------------------------------------------------------
# JSON-input adapter: load one or more exported-DB JSON dumps into an
# in-memory SQLite so the rest of the pipeline runs unchanged.
# ---------------------------------------------------------------------------

_MEM_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT, first_seen TEXT, last_seen TEXT,
    request_count INTEGER, cc_version TEXT, cch TEXT
);
CREATE TABLE requests (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT, api TEXT, role_kind TEXT, agent_label TEXT,
    model_requested TEXT, model_mapped TEXT,
    started_at TEXT, started_ms INTEGER,
    completed_at TEXT, completed_ms INTEGER, duration_ms INTEGER,
    status TEXT, status_code INTEGER,
    message_count INTEGER,
    prefix_hashes_json TEXT,
    request_body_json TEXT, converted_request_json TEXT,
    response_body_json TEXT, error_json TEXT,
    response_stop_reason TEXT,
    parent_trace_id TEXT, parent_agent_call_id TEXT,
    title TEXT
);
CREATE INDEX idx_req_session ON requests(session_id, started_ms);
CREATE TABLE agent_calls (
    agent_call_id TEXT PRIMARY KEY,
    session_id TEXT, parent_trace_id TEXT,
    tool_use_id TEXT, tool_name TEXT, agent_label TEXT,
    started_at TEXT, started_ms INTEGER,
    completed_at TEXT, completed_ms INTEGER, duration_ms INTEGER,
    status TEXT,
    input_preview TEXT, result_preview TEXT, result_trace_id TEXT
);
CREATE INDEX idx_ac_session ON agent_calls(session_id, started_ms);
"""


def _to_json_str(v: Any) -> Optional[str]:
    """Re-serialize dict/list back to JSON string for storage; pass through strings."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps(str(v), ensure_ascii=False)


_SESSION_COLS = ("session_id", "origin", "first_seen", "last_seen",
                 "request_count", "cc_version", "cch")
_REQ_COLS = (
    "trace_id", "session_id", "api", "role_kind", "agent_label",
    "model_requested", "model_mapped",
    "started_at", "started_ms",
    "completed_at", "completed_ms", "duration_ms",
    "status", "status_code", "message_count",
    "prefix_hashes_json", "request_body_json", "converted_request_json",
    "response_body_json", "error_json",
    "response_stop_reason",
    "parent_trace_id", "parent_agent_call_id", "title",
)
_AC_COLS = (
    "agent_call_id", "session_id", "parent_trace_id",
    "tool_use_id", "tool_name", "agent_label",
    "started_at", "started_ms", "completed_at", "completed_ms", "duration_ms",
    "status", "input_preview", "result_preview", "result_trace_id",
)
_JSON_FIELDS_IN_REQ = {
    "prefix_hashes_json", "request_body_json", "converted_request_json",
    "response_body_json", "error_json",
}


def _synthesize_agent_calls_from_tool_events(
    payload: Dict[str, Any], session_id: str
) -> List[Dict[str, Any]]:
    """For exports where agent_calls is empty but tool_events is populated,
    rebuild agent_call rows from Agent/Task tool_use_history events.  This is
    the fallback path for legacy dumps that predate the agent_calls resolver.

    Note: tool_use_history.event_time is the timestamp of the main-thread request
    that *carried* the tool_use block in its history, which can be much later than
    the actual subagent run.  We therefore back-fill started_ms from the earliest
    subagent request in the session, partitioning across multiple Agent calls by
    sorting both lists chronologically and pairing in order.
    """
    sub_starts = sorted(
        r.get("started_ms") for r in (payload.get("requests") or [])
        if isinstance(r, dict) and r.get("role_kind") == "subagent"
        and r.get("started_ms") is not None
    )
    agent_events = [
        te for te in (payload.get("tool_events") or [])
        if isinstance(te, dict)
        and te.get("event_type") == "tool_use_history"
        and (te.get("tool_name") or "").lower() in ("agent", "task")
        and te.get("tool_use_id")
    ]
    agent_events.sort(key=lambda e: e.get("event_time_ms") or 0)

    out = []
    for idx, te in enumerate(agent_events):
        tu_id = te["tool_use_id"]
        seed = f"{te.get('trace_id') or ''}|{tu_id}"
        agent_call_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:20]
        # Each Agent call gets the next "block" of subagent requests, partitioned
        # by index.  When there's only one Agent call, all subagent starts go to it.
        if len(agent_events) == 1:
            inferred_start = sub_starts[0] if sub_starts else te.get("event_time_ms")
        else:
            block_size = max(1, len(sub_starts) // len(agent_events))
            block_idx = idx * block_size
            inferred_start = (sub_starts[block_idx] if block_idx < len(sub_starts)
                              else te.get("event_time_ms"))
        out.append({
            "agent_call_id": agent_call_id,
            "session_id": session_id,
            "parent_trace_id": te.get("trace_id"),
            "tool_use_id": tu_id,
            "tool_name": te.get("tool_name"),
            "agent_label": te.get("agent_label"),
            "started_at": te.get("event_time"),
            "started_ms": inferred_start,
            "completed_at": None,
            "completed_ms": None,
            "duration_ms": None,
            "status": "unknown",
            "input_preview": te.get("input_preview"),
            "result_preview": None,
            "result_trace_id": None,
        })
    return out


def _ingest_json_payload(conn: sqlite3.Connection, payload: Any, src: str) -> int:
    """Insert one exported JSON dump into the in-memory db. Returns
    the number of sessions ingested from this payload."""
    # Accept either {session, requests, agent_calls, ...} or a list of such.
    if isinstance(payload, list):
        return sum(_ingest_json_payload(conn, item, src) for item in payload)
    if not isinstance(payload, dict):
        return 0

    sess = payload.get("session") or {}
    if not isinstance(sess, dict) or not sess.get("session_id"):
        print(f"  warning: {src} has no session_id, skipping", file=sys.stderr)
        return 0

    sess_row = tuple(sess.get(c) for c in _SESSION_COLS)
    conn.execute(
        f"INSERT OR REPLACE INTO sessions ({','.join(_SESSION_COLS)}) "
        f"VALUES ({','.join('?' * len(_SESSION_COLS))})",
        sess_row,
    )

    for r in payload.get("requests") or []:
        if not isinstance(r, dict):
            continue
        row = []
        for c in _REQ_COLS:
            v = r.get(c)
            if c in _JSON_FIELDS_IN_REQ:
                v = _to_json_str(v)
            row.append(v)
        conn.execute(
            f"INSERT OR REPLACE INTO requests ({','.join(_REQ_COLS)}) "
            f"VALUES ({','.join('?' * len(_REQ_COLS))})",
            tuple(row),
        )

    acs = payload.get("agent_calls") or []
    if not acs and (payload.get("tool_events") or []):
        synth = _synthesize_agent_calls_from_tool_events(payload, sess.get("session_id"))
        if synth:
            print(f"  note: {src} had no agent_calls; synthesized {len(synth)} from tool_events",
                  file=sys.stderr)
            acs = synth

    for ac in acs:
        if not isinstance(ac, dict):
            continue
        # Make sure session_id is set even if the dump omitted it on the row
        ac_local = dict(ac)
        ac_local.setdefault("session_id", sess.get("session_id"))
        row = tuple(ac_local.get(c) for c in _AC_COLS)
        conn.execute(
            f"INSERT OR REPLACE INTO agent_calls ({','.join(_AC_COLS)}) "
            f"VALUES ({','.join('?' * len(_AC_COLS))})",
            row,
        )
    return 1


def load_json_input(input_path: Path) -> sqlite3.Connection:
    """Build a temporary in-memory SQLite db from one JSON file or a directory
    of JSON files (one exported session per file)."""
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.json"))
        if not paths:
            raise FileNotFoundError(f"no *.json files found in {input_path}")
    else:
        paths = [input_path]

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_MEM_SCHEMA)

    total = 0
    for p in paths:
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warning: skipping {p} ({e})", file=sys.stderr)
            continue
        total += _ingest_json_payload(conn, payload, str(p))
    conn.commit()
    print(f"loaded {total} session(s) from {input_path} into in-memory db")
    return conn


RULE_FUNCS = {
    "failed_retry": rule_failed_retry,
    "dedupe_read": rule_dedupe_read,
    "compress_search": rule_compress_search,
    "overridden_write": rule_overridden_write,
}


def clean(messages: List[Dict[str, Any]], enabled_rules: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the cleaning passes until convergence. Return (kept_message_indices, deletions)."""
    steps = group_steps(messages)
    deletions: List[Dict[str, Any]] = []
    while True:
        total = 0
        for rule in enabled_rules:
            fn = RULE_FUNCS[rule]
            total += fn(messages, steps, deletions)
        if total == 0:
            break
    kept_indices = []
    for step in steps:
        if step.get("_deleted"):
            continue
        kept_indices.extend(step["message_indices"])
    return kept_indices, deletions


def _strip_internal_fields(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        clean_m = {k: v for k, v in m.items() if not k.startswith("_")}
        out.append(clean_m)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="cc_traces/trace.db",
                        help="Input: a trace.db SQLite file, OR an exported JSON dump "
                             "({session, requests, agent_calls, ...}), OR a directory of "
                             "such JSON files (one per session). Auto-detected by suffix.")
    parser.add_argument("--out", default="exports/training",
                        help="Output directory (default: exports/training)")
    parser.add_argument("--session", action="append",
                        help="Only export the given session_id(s).")
    parser.add_argument("--include-archived", action="store_true",
                        help="Also export archived sessions.")
    parser.add_argument("--raw-only", action="store_true",
                        help="Skip cleaning; only emit <session>.raw.json")
    parser.add_argument("--name-by", choices=("opsname", "session_id"), default="opsname",
                        help="Filename basis. 'opsname' extracts the Ascend operator name "
                             "from the task brief and falls back to session_id when not found "
                             "(default). 'session_id' uses the raw session id.")
    parser.add_argument("--export-subagents", action="store_true",
                        help="Also export each Agent/Task subagent run as its own training "
                             "sample <base>__sub<i>_<subagent_type>.{raw,clean,diff}.json.")
    for rule in ALL_RULES:
        parser.add_argument(f"--no-{rule.replace('_', '-')}", dest=f"rule_{rule}",
                            action="store_false",
                            help=f"Disable the {rule} cleaning rule.")
        parser.set_defaults(**{f"rule_{rule}": True})
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"error: trace db not found: {db_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    enabled_rules = [r for r in ALL_RULES if getattr(args, f"rule_{r}")]

    is_json_input = (
        db_path.is_dir() or
        db_path.suffix.lower() in (".json", ".jsonl")
    )
    if is_json_input:
        conn = load_json_input(db_path)
    else:
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
    session_rows = conn.execute(
        f"SELECT * FROM sessions {where_sql} ORDER BY last_seen DESC",
        params,
    ).fetchall()

    summary = {"total_sessions": 0, "written": 0, "skipped_no_main": 0,
               "total_msgs_raw": 0, "total_msgs_clean": 0,
               "named_by_opsname": 0, "named_by_session_fallback": 0,
               "sessions_with_compaction": 0, "total_compactions": 0,
               "subagents_written": 0, "subagent_msgs_raw": 0, "subagent_msgs_clean": 0,
               "deletions_by_rule": {r: 0 for r in enabled_rules}}

    used_names: Dict[str, int] = {}

    for s in session_rows:
        summary["total_sessions"] += 1
        bundle = build_raw_trajectory(conn, s)
        if not bundle or len(bundle["messages"]) < 2:
            summary["skipped_no_main"] += 1
            continue

        raw_messages = bundle["messages"]
        session_id = bundle["session_id"]
        short_sid = session_id.split("-")[0] if "-" in session_id else session_id[:8]

        opsname = extract_opsname(raw_messages) if args.name_by == "opsname" else None
        if opsname:
            base = opsname
            bundle["_meta"]["opsname"] = opsname
            summary["named_by_opsname"] += 1
        else:
            base = session_id.replace("/", "_")
            bundle["_meta"]["opsname"] = None
            if args.name_by == "opsname":
                summary["named_by_session_fallback"] += 1
        # disambiguate collisions (e.g. two sessions for the same op)
        count = used_names.get(base, 0)
        if count == 0:
            safe_name = base
        else:
            safe_name = f"{base}__{short_sid}"
        used_names[base] = count + 1

        main_safe_name = safe_name  # used to derive subagent filenames

        def _emit(bundle_, safe_, indent="  ", extra_tail=""):
            """Write raw/clean/diff for one bundle, return (raw_len, clean_len, n_drops, n_comp)."""
            raw_msgs = bundle_["messages"]
            n_comp_ = bundle_["_meta"]["n_compactions"]
            n_seg_ = bundle_["_meta"]["n_segments"]
            n_req_ = bundle_["_meta"]["n_requests"]
            comp_tag_ = f"COMPACTED×{n_comp_}" if n_comp_ else "no-compact"

            raw_out_ = {
                "session_id": bundle_["session_id"],
                "meta": bundle_["_meta"],
                "tools": bundle_["tools"],
                "messages": _strip_internal_fields(raw_msgs),
            }
            (out_dir / f"{safe_}.raw.json").write_text(
                json.dumps(raw_out_, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if args.raw_only:
                print(
                    f"{indent}{safe_:48s}  reqs={n_req_:3d}  segs={n_seg_}  "
                    f"{comp_tag_:14s}  msgs={len(raw_msgs):4d}{extra_tail}"
                )
                return len(raw_msgs), len(raw_msgs), 0, n_comp_

            kept_idx_, deletions_ = clean(raw_msgs, enabled_rules)
            clean_msgs_ = [raw_msgs[i] for i in kept_idx_]
            clean_out_ = {
                "session_id": bundle_["session_id"],
                "meta": {**bundle_["_meta"], "applied_rules": enabled_rules},
                "tools": bundle_["tools"],
                "messages": _strip_internal_fields(clean_msgs_),
            }
            (out_dir / f"{safe_}.clean.json").write_text(
                json.dumps(clean_out_, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            diff_out_ = {
                "session_id": bundle_["session_id"],
                "raw_message_count": len(raw_msgs),
                "clean_message_count": len(clean_msgs_),
                "dropped_message_count": len(raw_msgs) - len(clean_msgs_),
                "applied_rules": enabled_rules,
                "n_compactions": n_comp_,
                "deletions": deletions_,
            }
            (out_dir / f"{safe_}.diff.json").write_text(
                json.dumps(diff_out_, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            drops_here_: Dict[str, int] = {}
            for d in deletions_:
                summary["deletions_by_rule"][d["rule"]] = summary["deletions_by_rule"].get(d["rule"], 0) + 1
                drops_here_[d["rule"]] = drops_here_.get(d["rule"], 0) + 1
            drop_str_ = ",".join(f"{r}={n}" for r, n in drops_here_.items()) or "no-drops"
            print(
                f"{indent}{safe_:48s}  reqs={n_req_:3d}  segs={n_seg_}  "
                f"{comp_tag_:14s}  raw={len(raw_msgs):4d}→clean={len(clean_msgs_):4d}  "
                f"({drop_str_}){extra_tail}"
            )
            return len(raw_msgs), len(clean_msgs_), len(deletions_), n_comp_

        # ── main trajectory
        ops_tag = bundle["_meta"]["opsname"] or "(no opsname)"
        main_raw, main_clean, _, main_n_comp = _emit(
            bundle, main_safe_name, indent="  ",
            extra_tail=f"  sid={short_sid}  ops={ops_tag}",
        )
        summary["written"] += 1
        summary["total_msgs_raw"] += main_raw
        summary["total_msgs_clean"] += main_clean
        if main_n_comp:
            summary["sessions_with_compaction"] += 1
            summary["total_compactions"] += main_n_comp

        # ── subagent trajectories (opt-in)
        if args.export_subagents:
            sub_bundles = build_subagent_trajectories(conn, s)
            sub_count = len(sub_bundles)
            for sub_idx, (acid, sub_type, sub_bundle) in enumerate(sub_bundles, start=1):
                sub_type_safe = _FNAME_SAFE_RE.sub("_", sub_type or "subagent").strip("._-") or "subagent"
                suffix = f"sub{sub_idx}" if sub_count > 1 else "sub"
                sub_safe = f"{main_safe_name}__{suffix}_{sub_type_safe}"
                tail = f"  acid={acid[:8]}  type={sub_type}"
                s_raw, s_clean, _, s_comp = _emit(
                    sub_bundle, sub_safe, indent="    └─ ", extra_tail=tail
                )
                summary["subagents_written"] += 1
                summary["subagent_msgs_raw"] += s_raw
                summary["subagent_msgs_clean"] += s_clean
                if s_comp:
                    summary["total_compactions"] += s_comp

    print()
    print(f"sessions:  {summary['written']}/{summary['total_sessions']} written "
          f"(skipped {summary['skipped_no_main']} with no main thread)")
    if args.name_by == "opsname":
        print(f"naming:    {summary['named_by_opsname']} by opsname, "
              f"{summary['named_by_session_fallback']} fell back to session_id")
    if args.export_subagents:
        print(f"subagents: {summary['subagents_written']} written  "
              f"(msgs raw={summary['subagent_msgs_raw']} clean={summary['subagent_msgs_clean']})")
    print(f"compact:   {summary['sessions_with_compaction']} session(s) with /compact "
          f"({summary['total_compactions']} compactions across main+sub)")
    print(f"main msgs: raw={summary['total_msgs_raw']} clean={summary['total_msgs_clean']} "
          f"(-{summary['total_msgs_raw']-summary['total_msgs_clean']})")
    if not args.raw_only:
        print(f"drops:     {summary['deletions_by_rule']}")
    print(f"out:       {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
