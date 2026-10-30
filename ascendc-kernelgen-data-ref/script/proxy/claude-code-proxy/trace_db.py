"""SQLite-backed trace store for the claude-code proxy.

Designed to support analysis of Claude Code behavior under a mapped
backend (e.g. Qwen) for agentic-RL research. The schema records every
proxy /v1/messages and /v1/messages/count_tokens call, links them into
sessions, classifies them as titler / main / subagent, and extracts
tool events plus Agent-tool launches so the UI can show a clean picture
of the multi-agent flow.

Public surface (used by server.py and the UI handlers):

    record_request_started(...)
    record_request_completed(...)
    record_request_failed(...)
    clear_traces()
    list_sessions()
    get_session(session_id)
    list_requests(filters)
    get_request(trace_id)
    build_timeline(session_id)
    build_agent_tree(session_id)
    build_history_chains(session_id)
    snapshot_stats()
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRACE_DIR = Path(os.environ.get("CC_TRACE_DIR", "cc_traces"))
DB_PATH = TRACE_DIR / "trace.db"
TRACE_INCLUDE_SYSTEM = os.environ.get(
    "CC_TRACE_INCLUDE_SYSTEM_IN_PREFIX", "false"
).lower() in {"1", "true", "yes", "on"}

_DB_LOCK = threading.Lock()
_CONN_LOCAL = threading.local()


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

# Only redact actual credentials. The previous implementation matched any key
# containing "token", which clobbered usage.input_tokens / max_tokens. That made
# the trace useless for cost or behavior analysis.
_REDACT_KEYS = (
    "authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "api_key",
    "cookie",
    "set-cookie",
    "secret",
    "password",
    "credential",
)


def _is_secret_key(key: str) -> bool:
    key_lower = str(key).lower()
    return any(part == key_lower or part in key_lower for part in _REDACT_KEYS)


def sanitize(value: Any, key_hint: str = "") -> Any:
    if key_hint and _is_secret_key(key_hint):
        return "[redacted]"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, key_hint) for item in value]
    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump(mode="json"), key_hint)
    if hasattr(value, "dict"):
        return sanitize(value.dict(), key_hint)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def model_to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return sanitize(value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def iso_to_ms(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        normalized = str(value).replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except Exception:
        return None


def monotonic_ms(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)


def new_trace_id(prefix: str = "cc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any, length: int = 20) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()[:length]


def headers_to_dict(headers: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    return {key: sanitize(value, key) for key, value in headers}


_trace_enabled_override: Optional[bool] = None


def is_trace_enabled() -> bool:
    if _trace_enabled_override is not None:
        return _trace_enabled_override
    return os.environ.get("CC_TRACE_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def set_trace_enabled(enabled: bool) -> None:
    global _trace_enabled_override
    _trace_enabled_override = enabled


def get_trace_enabled() -> bool:
    return is_trace_enabled()


def ensure_trace_dir() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_text(value: Any) -> str:
    pieces: List[str] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            pieces.append(item)
            return
        if isinstance(item, dict):
            t = item.get("type")
            if t == "text" and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            elif "text" in item and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            elif t in {"tool_use", "tool_result"}:
                pieces.append(stable_json(item))
            else:
                for v in item.values():
                    walk(v)
            return
        if isinstance(item, list):
            for v in item:
                walk(v)
            return

    walk(value)
    return "\n".join(p for p in pieces if p)


def compact_preview(text: str, max_chars: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def block_types(content: Any) -> List[str]:
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                out.append(str(b.get("type", "object")))
            else:
                out.append(type(b).__name__)
        return sorted(set(out))
    if isinstance(content, str):
        return ["text"]
    if isinstance(content, dict):
        return [str(content.get("type", "object"))]
    return [type(content).__name__]


def normalize_for_hash(value: Any) -> Any:
    """Stable canonical form for prefix-equality hashing."""
    if isinstance(value, dict):
        return {str(k): normalize_for_hash(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_hash(item) for item in value]
    return value


def _canonical_text_for_approx(value: Any, max_chars: int = 800) -> str:
    text = extract_text(value)
    if not text:
        try:
            text = stable_json(value)
        except Exception:
            text = str(value)
    text = re.sub(r"call_[0-9a-fA-F]+", "call_*", text)
    text = re.sub(r"toolu_[A-Za-z0-9_-]+", "toolu_*", text)
    text = re.sub(r"chatcmpl-[A-Za-z0-9_-]+", "chatcmpl_*", text)
    text = re.sub(r"\b[0-9a-f]{16,}\b", "hash_*", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def normalize_message_for_approx(role: str, content: Any) -> Dict[str, Any]:
    if isinstance(content, list):
        norm: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type", "object")
            if t == "text":
                norm.append({"t": "text", "x": _canonical_text_for_approx(block.get("text"))})
            elif t == "tool_use":
                norm.append(
                    {
                        "t": "tool_use",
                        "n": block.get("name"),
                        "i": _canonical_text_for_approx(block.get("input")),
                    }
                )
            elif t == "tool_result":
                norm.append(
                    {
                        "t": "tool_result",
                        "x": _canonical_text_for_approx(block.get("content")),
                    }
                )
            else:
                norm.append({"t": t, "x": _canonical_text_for_approx(block)})
        return {"role": role, "blocks": norm}
    return {"role": role, "blocks": [{"t": "text", "x": _canonical_text_for_approx(content)}]}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

_TITLER_HINTS = (
    "concise, sentence-case title",
    "Generate a concise",
    "summarize the session",
)

# Heuristic regexes scraped from observed CC system prompts.
_CC_VERSION_RE = re.compile(r"cc_version=([0-9A-Za-z._\-]+)")
_BILLING_HEADER_RE = re.compile(r"x-anthropic-billing-header:.*?cch=([0-9A-Za-z]+)")


def _system_text_of(body: Dict[str, Any]) -> str:
    sys = body.get("system")
    if isinstance(sys, str):
        return sys
    if isinstance(sys, list):
        parts: List[str] = []
        for s in sys:
            if isinstance(s, dict):
                t = s.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def classify_request(api: str, body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Decide whether this is a title-generator, main agent, or subagent call."""
    if api != "messages":
        return {
            "role_kind": "system",
            "agent_label": api,
            "cc_version": None,
            "cch": None,
        }

    sys_text = _system_text_of(body)
    tools = body.get("tools") or []
    tool_names = [t.get("name") for t in tools if isinstance(t, dict) and t.get("name")]
    has_agent_tool = any(n in {"Agent", "Task"} for n in tool_names)

    cc_version_match = _CC_VERSION_RE.search(sys_text)
    cch_match = _BILLING_HEADER_RE.search(sys_text)

    role_kind: str
    agent_label: str
    if any(hint in sys_text for hint in _TITLER_HINTS) and len(tool_names) == 0:
        role_kind = "titler"
        agent_label = "titler"
    # --- Subagent identification (version-compatible, highest priority first) ---
    # L1: CC >= 2.1.221 injects cc_is_subagent=true into the billing-header system
    #     block. Authoritative signal — checked BEFORE the tools heuristic because
    #     221+ subagents advertise the full tool list (incl. Agent), which used to
    #     misclassify them as main.
    elif "cc_is_subagent=true" in sys_text:
        role_kind = "subagent"
        agent_label = "subagent"
    # L2: subagent-only system prose ("You are an agent for Claude Code...").
    #     Version-independent; covers CC builds and SDK subagents that lack the
    #     billing-header marker but still carry Agent in their tool list.
    elif "You are an agent for Claude Code" in sys_text:
        role_kind = "subagent"
        agent_label = "subagent"
    elif has_agent_tool:
        role_kind = "main"
        agent_label = "main"
    elif len(tool_names) > 0:
        # Has tools but no Agent-tool advertisement → likely a Task subagent.
        # Subagent type can be deduced later from the parent's Agent tool_use input.
        role_kind = "subagent"
        agent_label = "subagent"
    else:
        # No tools, no Agent ad, no titler hint → an external/raw client call
        # (e.g. a plain Claude Agent SDK invocation that does not mimic CC).
        role_kind = "external"
        agent_label = "external"

    return {
        "role_kind": role_kind,
        "agent_label": agent_label,
        "cc_version": cc_version_match.group(1) if cc_version_match else None,
        "cch": cch_match.group(1) if cch_match else None,
    }


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL DEFAULT 'header',     -- 'header' | 'derived'
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    cc_version TEXT,
    cch TEXT,
    archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS requests (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT,
    api TEXT NOT NULL,
    role_kind TEXT NOT NULL,                  -- 'titler' | 'main' | 'subagent' | 'external' | 'system'
    agent_label TEXT NOT NULL,
    model_requested TEXT,
    model_mapped TEXT,
    started_at TEXT NOT NULL,
    started_ms INTEGER,
    completed_at TEXT,
    completed_ms INTEGER,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'inflight',  -- 'success' | 'error' | 'inflight'
    status_code INTEGER,
    method TEXT,
    path TEXT,
    client TEXT,
    pid INTEGER,
    cc_version TEXT,
    cch TEXT,
    headers_json TEXT,
    system_text TEXT,
    system_chars INTEGER,
    system_hash TEXT,
    tool_count INTEGER NOT NULL DEFAULT 0,
    tool_names_json TEXT,
    advertises_agent_tool INTEGER NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    history_hash TEXT,
    history_hash_approx TEXT,
    prefix_hashes_json TEXT,
    request_body_json TEXT,
    converted_request_json TEXT,
    response_body_json TEXT,
    error_json TEXT,
    extra_json TEXT,
    response_stop_reason TEXT,
    response_text TEXT,
    response_tool_uses_json TEXT,
    response_usage_json TEXT,
    parent_trace_id TEXT,
    parent_match_kind TEXT,
    parent_added_steps INTEGER,
    parent_agent_call_id TEXT,
    title TEXT
);

CREATE INDEX IF NOT EXISTS idx_req_session ON requests(session_id, started_ms);
CREATE INDEX IF NOT EXISTS idx_req_history ON requests(history_hash);
CREATE INDEX IF NOT EXISTS idx_req_role ON requests(role_kind);
CREATE INDEX IF NOT EXISTS idx_req_started ON requests(started_ms);

CREATE TABLE IF NOT EXISTS messages (
    trace_id TEXT NOT NULL,
    msg_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    role_or_source TEXT,                      -- 'system' | 'user' | 'assistant' | 'response'
    content_kind TEXT,                        -- 'text' | 'blocks'
    text_preview TEXT,
    full_content_json TEXT,
    canonical_hash TEXT,
    chars INTEGER,
    block_types_json TEXT,
    PRIMARY KEY (trace_id, msg_index)
);
CREATE INDEX IF NOT EXISTS idx_msg_hash ON messages(canonical_hash);

CREATE TABLE IF NOT EXISTS tool_events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    session_id TEXT,
    location TEXT NOT NULL,                   -- 'request_history' | 'response'
    event_type TEXT NOT NULL,                 -- 'tool_use_history' | 'tool_result' | 'tool_use_response'
    message_index INTEGER,
    block_index INTEGER,
    role TEXT,
    tool_name TEXT,
    tool_use_id TEXT,                         -- for tool_use_*; for tool_result this mirrors tool_result_id
    tool_result_id TEXT,                      -- only for tool_result events
    agent_label TEXT,
    event_time TEXT NOT NULL,
    event_time_ms INTEGER,
    input_preview TEXT,
    raw_block_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_te_trace ON tool_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_te_use_id ON tool_events(tool_use_id);
CREATE INDEX IF NOT EXISTS idx_te_result_id ON tool_events(tool_result_id);
CREATE INDEX IF NOT EXISTS idx_te_time ON tool_events(event_time_ms);
CREATE INDEX IF NOT EXISTS idx_te_session ON tool_events(session_id);

CREATE TABLE IF NOT EXISTS agent_calls (
    agent_call_id TEXT PRIMARY KEY,
    parent_trace_id TEXT NOT NULL,
    session_id TEXT,
    tool_use_id TEXT,
    tool_name TEXT,
    agent_label TEXT,                         -- subagent_type ('general-purpose', 'Explore', ...)
    started_at TEXT,
    started_ms INTEGER,
    completed_at TEXT,
    completed_ms INTEGER,
    duration_ms INTEGER,
    status TEXT,                              -- 'open' | 'completed'
    input_preview TEXT,
    result_preview TEXT,
    result_trace_id TEXT,
    child_request_ids_json TEXT,
    history_request_ids_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_ac_parent ON agent_calls(parent_trace_id);
CREATE INDEX IF NOT EXISTS idx_ac_use_id ON agent_calls(tool_use_id);
CREATE INDEX IF NOT EXISTS idx_ac_session ON agent_calls(session_id);
"""


def _conn() -> sqlite3.Connection:
    conn = getattr(_CONN_LOCAL, "conn", None)
    if conn is not None:
        return conn
    ensure_trace_dir()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    _CONN_LOCAL.conn = conn
    return conn


# ---------------------------------------------------------------------------
# Per-operator DB routing (按算子分库)
#
# 每个 session_id 映射到一个 op_key（如 "41_ELU"），该 session 的所有读写
# 路由到 {op_key}.db。无映射的 session 回退到默认 trace.db。
# 跨 db 查询（list_*, _session_for_trace 等）通过 _all_conns() 遍历所有 db。
# ---------------------------------------------------------------------------

# session_id → op_key 内存映射
_SESSION_OP_MAP: Dict[str, str] = {}
_SESSION_OP_LOCK = threading.Lock()

# 每线程的 op_key → Connection 缓存（同 trace.db 的 _CONN_LOCAL 思路）
_OP_CONN_LOCAL = threading.local()


def set_session_op(session_id: str, op_key: str) -> None:
    """建立 session_id → op_key 映射。后续该 session 的读写路由到 {op_key}.db"""
    with _SESSION_OP_LOCK:
        if _SESSION_OP_MAP.get(session_id) != op_key:
            _SESSION_OP_MAP[session_id] = op_key


def get_session_op(session_id: str) -> Optional[str]:
    """查询 session_id 对应的 op_key，无映射返回 None（用默认 trace.db）"""
    with _SESSION_OP_LOCK:
        return _SESSION_OP_MAP.get(session_id)


def clear_session_op_map() -> None:
    """清空 session_id → op_key 映射（测试或重置用）"""
    with _SESSION_OP_LOCK:
        _SESSION_OP_MAP.clear()


def _op_conn(op_key: str) -> sqlite3.Connection:
    """获取/创建 op_key 对应的连接（每线程缓存，lazy 创建）"""
    conns = getattr(_OP_CONN_LOCAL, "conns", None)
    if conns is None:
        conns = {}
        _OP_CONN_LOCAL.conns = conns
    conn = conns.get(op_key)
    if conn is not None:
        return conn
    ensure_trace_dir()
    db_path = TRACE_DIR / f"{op_key}.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conns[op_key] = conn
    return conn


def _conn_for_session(session_id: Optional[str]) -> sqlite3.Connection:
    """按 session_id 选 db 连接。有 op_key 映射则用 {op_key}.db，否则默认 trace.db"""
    if session_id:
        op_key = get_session_op(session_id)
        if op_key:
            return _op_conn(op_key)
    return _conn()


def _all_db_paths() -> List[Path]:
    """列出 TRACE_DIR 下所有 .db 文件（trace.db + 各 {op_key}.db）"""
    ensure_trace_dir()
    return sorted(TRACE_DIR.glob("*.db"))


def _all_conns() -> List[sqlite3.Connection]:
    """获取所有 db 的连接（默认 trace.db + 所有 {op_key}.db），用于跨 db 查询"""
    conns: List[sqlite3.Connection] = [_conn()]
    seen = {"trace.db"}
    # 内存里已缓存的 op 连接
    op_conns = getattr(_OP_CONN_LOCAL, "conns", None) or {}
    for op_key, conn in op_conns.items():
        conns.append(conn)
        seen.add(f"{op_key}.db")
    # 磁盘上存在但还没在内存缓存的 db（proxy 重启后，或别处创建的）
    for db_path in _all_db_paths():
        if db_path.name in seen or db_path.name == "trace.db":
            continue
        conns.append(_op_conn(db_path.stem))
    return conns


def clear_traces() -> None:
    with _DB_LOCK:
        for c in _all_conns():
            c.execute("DELETE FROM tool_events")
            c.execute("DELETE FROM agent_calls")
            c.execute("DELETE FROM messages")
            c.execute("DELETE FROM requests")
            c.execute("DELETE FROM sessions")


# ---------------------------------------------------------------------------
# Recording: request_started → row insert with status='inflight'
# ---------------------------------------------------------------------------

def _per_message_canonical(role: str, content: Any) -> Dict[str, Any]:
    canonical = {"role": role, "content": normalize_for_hash(content)}
    return {
        "hash": stable_hash(canonical),
        "preview": compact_preview(extract_text(content)) if content else "",
        "chars": len(extract_text(content)) if content else 0,
        "block_types": block_types(content),
        "kind": "blocks" if isinstance(content, list) else ("text" if isinstance(content, str) else "object"),
    }


def _store_messages(c: sqlite3.Connection, trace_id: str, body: Dict[str, Any]) -> List[str]:
    rows: List[Tuple] = []
    hashes: List[str] = []

    if TRACE_INCLUDE_SYSTEM and body.get("system") is not None:
        canonical = {"role": "system", "content": normalize_for_hash(body.get("system"))}
        h = stable_hash(canonical)
        hashes.append(h)
        rows.append(
            (
                trace_id,
                -1,
                "system",
                "system",
                "blocks" if isinstance(body.get("system"), list) else "text",
                compact_preview(extract_text(body.get("system"))),
                json.dumps(sanitize(body.get("system")), ensure_ascii=False, sort_keys=True),
                h,
                len(extract_text(body.get("system"))),
                json.dumps(block_types(body.get("system"))),
            )
        )

    for idx, msg in enumerate(body.get("messages") or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content")
        meta = _per_message_canonical(role, content)
        hashes.append(meta["hash"])
        rows.append(
            (
                trace_id,
                idx,
                role,
                role,
                meta["kind"],
                meta["preview"],
                json.dumps(sanitize(content), ensure_ascii=False, sort_keys=True),
                meta["hash"],
                meta["chars"],
                json.dumps(meta["block_types"]),
            )
        )

    if rows:
        c.executemany(
            """
            INSERT OR REPLACE INTO messages
            (trace_id, msg_index, role, role_or_source, content_kind, text_preview,
             full_content_json, canonical_hash, chars, block_types_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    return hashes


def _approx_history_hash(body: Dict[str, Any]) -> str:
    norm = []
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            norm.append(normalize_message_for_approx(msg.get("role", "unknown"), msg.get("content")))
    return stable_hash({"approx": norm})


def _extract_response_signals(response: Any) -> Dict[str, Any]:
    if not isinstance(response, dict):
        return {"text": "", "tool_uses": [], "stop_reason": None, "usage": None}
    text_parts: List[str] = []
    tool_uses: List[Dict[str, Any]] = []
    for b in response.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text_parts.append(b.get("text") or "")
        elif b.get("type") == "tool_use":
            tool_uses.append(
                {
                    "id": b.get("id"),
                    "name": b.get("name"),
                    "input_preview": compact_preview(extract_text(b.get("input")), 200),
                    "input": sanitize(b.get("input")),
                }
            )
    return {
        "text": "\n".join(p for p in text_parts if p).strip(),
        "tool_uses": tool_uses,
        "stop_reason": response.get("stop_reason"),
        "usage": response.get("usage"),
    }


def _extract_title(response_text: str) -> Optional[str]:
    if not response_text:
        return None
    text = response_text.strip()
    # Try parsing a {"title": "..."} JSON object.
    try:
        # Accept either trailing or leading text around JSON.
        match = re.search(r"\{.*?\}", text, re.S)
        if match:
            obj = json.loads(match.group(0))
            t = obj.get("title")
            if isinstance(t, str):
                return t.strip()
    except Exception:
        pass
    # Fallback: first non-empty line.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return None


def archive_session(session_id: str) -> bool:
    c = _conn_for_session(session_id)
    cur = c.execute(
        "UPDATE sessions SET archived = 1 WHERE session_id = ?", (session_id,)
    )
    return cur.rowcount > 0


def unarchive_session(session_id: str) -> bool:
    c = _conn_for_session(session_id)
    cur = c.execute(
        "UPDATE sessions SET archived = 0 WHERE session_id = ?", (session_id,)
    )
    return cur.rowcount > 0


def _upsert_session(c: sqlite3.Connection, session_id: Optional[str], started_at: str,
                    cc_version: Optional[str], cch: Optional[str],
                    origin: str = "header") -> None:
    if not session_id:
        return
    c.execute(
        """
        INSERT INTO sessions(session_id, origin, first_seen, last_seen, request_count, cc_version, cch)
        VALUES (?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            last_seen = max(excluded.last_seen, sessions.last_seen),
            first_seen = min(excluded.first_seen, sessions.first_seen),
            request_count = sessions.request_count + 1,
            cc_version = COALESCE(sessions.cc_version, excluded.cc_version),
            cch = COALESCE(sessions.cch, excluded.cch)
        """,
        (session_id, origin, started_at, started_at, cc_version, cch),
    )


# 批跑脚本 PROMPT 里的"算子描述文件为 <path>.py"模式，用于按算子分库
_OP_FILE_PATTERN = re.compile(r"算子描述文件为\s+(\S+\.py)")


def _extract_op_key_from_body(body: Dict[str, Any]) -> Optional[str]:
    """从请求 body 解析算子描述文件路径，返回 op_key（basename 去后缀）。

    遍历 messages 找 user 消息里的"算子描述文件为 <path>.py"。
    """
    messages = body.get("messages") or []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
        # 找所有匹配，跳过占位符示例（/path/to/...），取第一个真实算子路径
        for m in _OP_FILE_PATTERN.finditer(text):
            path = m.group(1).strip().rstrip(",，")
            if "/path/to/" in path:
                continue
            try:
                return Path(path).stem
            except Exception:
                name = path.rsplit("/", 1)[-1]
                return name.rsplit(".", 1)[0] if "." in name else name
    return None


def record_request_started(
    *,
    trace_id: str,
    api: str,
    method: str,
    path: str,
    headers: Dict[str, str],
    client: Optional[str],
    body_json: Dict[str, Any],
    mapped_model: Optional[str] = None,
) -> None:
    if not is_trace_enabled():
        return

    started_at = utc_now_iso()
    started_ms = iso_to_ms(started_at)

    body_clean = sanitize(body_json) or {}
    cls = classify_request(api, body_clean, headers or {})
    role_kind = cls["role_kind"]
    agent_label = cls["agent_label"]

    sys_text = _system_text_of(body_clean)
    sys_hash = stable_hash(body_clean.get("system")) if body_clean.get("system") is not None else None

    tools = body_clean.get("tools") or []
    tool_names = sorted({t.get("name") for t in tools if isinstance(t, dict) and t.get("name")})
    advertises_agent_tool = int(any(n in {"Agent", "Task"} for n in tool_names))

    session_id = (headers or {}).get("x-claude-code-session-id") or (headers or {}).get(
        "x-claude-session-id"
    )
    if not session_id:
        meta = body_clean.get("metadata") or {}
        if isinstance(meta, dict):
            user_id_blob = meta.get("user_id")
            if isinstance(user_id_blob, str):
                try:
                    parsed = json.loads(user_id_blob)
                    if isinstance(parsed, dict):
                        session_id = parsed.get("session_id") or session_id
                except Exception:
                    pass

    # Fallback grouping for non-Claude-Code clients (raw SDK / curl / our own
    # smoke tests): derive a stable bucket from client identity + UTC day so
    # those requests still appear under a session row in the UI.
    session_origin = "header" if session_id else None
    if not session_id:
        ua_raw = (headers or {}).get("user-agent") or "unknown-ua"
        # Path-safe shortening: keep alnum/dot/dash; collapse anything else.
        ua = re.sub(r"[^A-Za-z0-9._-]+", "-", ua_raw)[:60].strip("-") or "ua"
        day = started_at[:10]
        client_safe = re.sub(r"[^A-Za-z0-9._-]+", "-", client or "unknown-client")
        session_id = f"ext-{client_safe}-{ua}-{day}"
        session_origin = "derived"

    # 按算子分库：如果是新 session，解析 prompt 拿 op_key，建立映射
    if session_id and not get_session_op(session_id):
        op_key = _extract_op_key_from_body(body_clean)
        if op_key:
            set_session_op(session_id, op_key)

    prefix_hashes: List[str] = []
    history_hash: Optional[str] = None
    history_hash_approx: Optional[str] = None

    with _DB_LOCK:
        c = _conn_for_session(session_id)
        c.execute("BEGIN")
        try:
            prefix_hashes = _store_messages(c, trace_id, body_clean)
            history_hash = stable_hash({"prefix": prefix_hashes})
            history_hash_approx = _approx_history_hash(body_clean)

            _upsert_session(c, session_id, started_at, cls.get("cc_version"), cls.get("cch"), origin=session_origin or "header")

            c.execute(
                """
                INSERT OR REPLACE INTO requests
                (trace_id, session_id, api, role_kind, agent_label, model_requested, model_mapped,
                 started_at, started_ms, status, method, path, client, pid, cc_version, cch,
                 headers_json, system_text, system_chars, system_hash,
                 tool_count, tool_names_json, advertises_agent_tool,
                 message_count, history_hash, history_hash_approx, prefix_hashes_json,
                 request_body_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'inflight', ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    session_id,
                    api,
                    role_kind,
                    agent_label,
                    body_clean.get("model"),
                    mapped_model,
                    started_at,
                    started_ms,
                    method,
                    path,
                    client,
                    os.getpid(),
                    cls.get("cc_version"),
                    cls.get("cch"),
                    json.dumps(headers or {}, ensure_ascii=False, sort_keys=True),
                    sys_text,
                    len(sys_text),
                    sys_hash,
                    len(tool_names),
                    json.dumps(tool_names),
                    advertises_agent_tool,
                    len(body_clean.get("messages") or []),
                    history_hash,
                    history_hash_approx,
                    json.dumps(prefix_hashes),
                    json.dumps(body_clean, ensure_ascii=False, sort_keys=True),
                ),
            )
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise


def _store_tool_events(c: sqlite3.Connection, trace_id: str, session_id: Optional[str],
                       request_body: Dict[str, Any], response: Any,
                       started_at: str, completed_at: Optional[str]) -> None:
    """Re-derive and persist tool_use / tool_result events for this request."""
    c.execute("DELETE FROM tool_events WHERE trace_id = ?", (trace_id,))

    rows: List[Tuple] = []
    # Within this request, track tool_use_id -> (tool_name, agent_label) so that
    # tool_result blocks for the same id inherit the correct tool/agent label.
    use_meta: Dict[str, Tuple[str, str]] = {}

    def add(event_type: str, location: str, message_index: Optional[int], block_index: int,
            role: str, block: Dict[str, Any], event_time: str) -> None:
        block_type = block.get("type")
        tool_name = None
        tool_use_id = None
        tool_result_id = None
        agent_label = "main"
        preview_value: Any = None
        if block_type == "tool_use":
            tool_name = str(block.get("name") or "") or None
            tool_use_id = str(block.get("id") or "") or None
            tinput = block.get("input")
            if isinstance(tinput, dict):
                for key in ("subagent_type", "agent_type", "agent", "subagent"):
                    v = tinput.get(key)
                    if isinstance(v, str) and v.strip():
                        agent_label = v.strip()
                        break
            if tool_use_id:
                use_meta[tool_use_id] = (tool_name or "", agent_label)
            preview_value = tinput
        elif block_type == "tool_result":
            tool_result_id = str(block.get("tool_use_id") or "") or None
            tool_use_id = tool_result_id
            preview_value = block.get("content")
            if tool_result_id and tool_result_id in use_meta:
                tool_name, agent_label = use_meta[tool_result_id]
        else:
            return

        event_id = stable_hash(
            {
                "trace_id": trace_id,
                "event_type": event_type,
                "location": location,
                "message_index": message_index,
                "block_index": block_index,
                "tool_use_id": tool_use_id,
                "tool_result_id": tool_result_id,
            }
        )
        rows.append(
            (
                event_id,
                trace_id,
                session_id,
                location,
                event_type,
                message_index,
                block_index,
                role,
                tool_name,
                tool_use_id,
                tool_result_id,
                agent_label,
                event_time,
                iso_to_ms(event_time),
                compact_preview(extract_text(preview_value), 320) if preview_value is not None else "",
                json.dumps(sanitize(block), ensure_ascii=False, sort_keys=True),
            )
        )

    messages = (request_body or {}).get("messages") or []
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "tool_use":
                add("tool_use_history", "request_history", mi, bi, role, block, started_at)
            elif t == "tool_result":
                add("tool_result", "request_history", mi, bi, role, block, started_at)

    if isinstance(response, dict):
        for bi, block in enumerate(response.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                add(
                    "tool_use_response",
                    "response",
                    None,
                    bi,
                    "assistant",
                    block,
                    completed_at or started_at,
                )

    if rows:
        c.executemany(
            """
            INSERT OR REPLACE INTO tool_events
            (event_id, trace_id, session_id, location, event_type, message_index, block_index,
             role, tool_name, tool_use_id, tool_result_id, agent_label, event_time,
             event_time_ms, input_preview, raw_block_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def record_request_completed(
    *,
    trace_id: str,
    status_code: int,
    duration_ms: int,
    converted_request: Optional[Dict[str, Any]] = None,
    response: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> None:
    if not is_trace_enabled():
        return

    completed_at = utc_now_iso()
    completed_ms = iso_to_ms(completed_at)
    response_plain = model_to_plain(response) if response is not None else None
    response_clean = sanitize(response_plain)
    converted_clean = sanitize(converted_request or {})

    signals = _extract_response_signals(response_clean)
    title = _extract_title(signals.get("text", "")) if signals.get("text") else None

    if session_id is None:
        session_id = _session_for_trace(trace_id)

    with _DB_LOCK:
        c = _conn_for_session(session_id)
        c.execute("BEGIN")
        try:
            row = c.execute(
                "SELECT request_body_json, started_at, session_id, role_kind FROM requests WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if not row:
                c.execute("ROLLBACK")
                return
            body = json.loads(row["request_body_json"]) if row["request_body_json"] else {}
            session_id = row["session_id"]

            c.execute(
                """
                UPDATE requests
                SET status = 'success',
                    status_code = ?,
                    completed_at = ?,
                    completed_ms = ?,
                    duration_ms = ?,
                    converted_request_json = ?,
                    response_body_json = ?,
                    extra_json = ?,
                    response_stop_reason = ?,
                    response_text = ?,
                    response_tool_uses_json = ?,
                    response_usage_json = ?,
                    title = COALESCE(?, title)
                WHERE trace_id = ?
                """,
                (
                    status_code,
                    completed_at,
                    completed_ms,
                    duration_ms,
                    json.dumps(converted_clean, ensure_ascii=False, sort_keys=True),
                    json.dumps(response_clean, ensure_ascii=False, sort_keys=True) if response_clean is not None else None,
                    json.dumps(extra or {}, ensure_ascii=False, sort_keys=True),
                    signals.get("stop_reason"),
                    signals.get("text"),
                    json.dumps(signals.get("tool_uses") or [], ensure_ascii=False),
                    json.dumps(signals.get("usage") or {}, ensure_ascii=False),
                    title,
                    trace_id,
                ),
            )
            _store_tool_events(c, trace_id, session_id, body, response_clean,
                               row["started_at"], completed_at)
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    # Continuity / agent-call resolution can run outside the write lock.
    _resolve_continuity(trace_id)
    _resolve_agent_calls_for_session(_session_for_trace(trace_id))


def record_request_failed(
    *,
    trace_id: str,
    status_code: int,
    duration_ms: int,
    converted_request: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> None:
    if not is_trace_enabled():
        return

    completed_at = utc_now_iso()
    completed_ms = iso_to_ms(completed_at)
    converted_clean = sanitize(converted_request or {})
    error_clean = sanitize(error or {})

    if session_id is None:
        session_id = _session_for_trace(trace_id)

    with _DB_LOCK:
        c = _conn_for_session(session_id)
        c.execute(
            """
            UPDATE requests
            SET status = 'error',
                status_code = ?,
                completed_at = ?,
                completed_ms = ?,
                duration_ms = ?,
                converted_request_json = ?,
                error_json = ?
            WHERE trace_id = ?
            """,
            (
                status_code,
                completed_at,
                completed_ms,
                duration_ms,
                json.dumps(converted_clean, ensure_ascii=False, sort_keys=True),
                json.dumps(error_clean, ensure_ascii=False, sort_keys=True),
                trace_id,
            ),
        )


# ---------------------------------------------------------------------------
# Continuity resolution
# ---------------------------------------------------------------------------

def _session_for_trace(trace_id: str) -> Optional[str]:
    for c in _all_conns():
        row = c.execute("SELECT session_id FROM requests WHERE trace_id = ?", (trace_id,)).fetchone()
        if row:
            return row["session_id"]
    return None


def _resolve_continuity(trace_id: str) -> None:
    """Attach parent_trace_id by looking for any earlier request whose history is a strict prefix.

    Falls back to approximate-prefix match (tail drift up to 2 messages, normalized hashing).
    """
    session_id = _session_for_trace(trace_id)
    if session_id is None:
        return
    c = _conn_for_session(session_id)
    row = c.execute(
        "SELECT session_id, started_ms, prefix_hashes_json, history_hash_approx, message_count "
        "FROM requests WHERE trace_id = ?",
        (trace_id,),
    ).fetchone()
    if not row:
        return
    started_ms = row["started_ms"] or 0
    if not row["prefix_hashes_json"]:
        return
    child_hashes = json.loads(row["prefix_hashes_json"])
    if not child_hashes:
        return

    # candidates: same session (or both null), earlier in time
    where = "session_id IS ?" if session_id is None else "session_id = ?"
    candidates = c.execute(
        f"""
        SELECT trace_id, prefix_hashes_json, started_ms, role_kind, message_count
        FROM requests
        WHERE {where}
          AND trace_id != ?
          AND started_ms IS NOT NULL
          AND started_ms <= ?
        ORDER BY started_ms ASC
        """,
        (session_id, trace_id, started_ms),
    ).fetchall()

    best: Optional[Dict[str, Any]] = None
    best_score: Tuple[int, int, int, int] = (0, 0, 0, 0)
    for cand in candidates:
        cand_hashes = json.loads(cand["prefix_hashes_json"]) if cand["prefix_hashes_json"] else []
        if not cand_hashes or len(cand_hashes) >= len(child_hashes):
            continue
        common = 0
        for a, b in zip(cand_hashes, child_hashes):
            if a == b:
                common += 1
            else:
                break
        if common == len(cand_hashes):
            score = (2, common, -1, cand["started_ms"] or 0)
            if score > best_score:
                best_score = score
                best = {
                    "parent_trace_id": cand["trace_id"],
                    "match_kind": "strict_prefix",
                    "added_steps": len(child_hashes) - len(cand_hashes),
                }
        elif common >= max(1, len(cand_hashes) - 2) and common >= 2:
            tail_drift = len(cand_hashes) - common
            score = (1, common, -tail_drift, cand["started_ms"] or 0)
            if score > best_score:
                best_score = score
                best = {
                    "parent_trace_id": cand["trace_id"],
                    "match_kind": f"tail_drift_{tail_drift}",
                    "added_steps": len(child_hashes) - common,
                }

    if best:
        c.execute(
            "UPDATE requests SET parent_trace_id = ?, parent_match_kind = ?, parent_added_steps = ? WHERE trace_id = ?",
            (best["parent_trace_id"], best["match_kind"], best["added_steps"], trace_id),
        )
    else:
        c.execute(
            "UPDATE requests SET parent_trace_id = NULL, parent_match_kind = NULL, parent_added_steps = NULL WHERE trace_id = ?",
            (trace_id,),
        )


def _resolve_agent_calls_for_session(session_id: Optional[str]) -> None:
    """Pair Agent/Task tool_use_response events with their tool_result + child requests."""
    c = _conn_for_session(session_id)
    where = "session_id IS ?" if session_id is None else "session_id = ?"

    starts = c.execute(
        f"""
        SELECT * FROM tool_events
        WHERE {where}
          AND event_type = 'tool_use_response'
          AND (lower(tool_name) IN ('agent','task') OR agent_label != 'main')
        ORDER BY event_time_ms ASC, event_id ASC
        """,
        (session_id,),
    ).fetchall()

    # Pull all tool_result events for this session — we will pair them by tool_use_id
    # below. We do not filter by tool_name here because tool_result blocks do not
    # carry the originating tool's name, only its id.
    results = c.execute(
        f"""
        SELECT * FROM tool_events
        WHERE {where}
          AND event_type = 'tool_result'
        ORDER BY event_time_ms ASC, event_id ASC
        """,
        (session_id,),
    ).fetchall()

    requests = c.execute(
        f"""
        SELECT trace_id, started_ms, role_kind FROM requests
        WHERE {where}
          AND api = 'messages'
        ORDER BY started_ms ASC
        """,
        (session_id,),
    ).fetchall()

    results_by_id: Dict[str, List[sqlite3.Row]] = {}
    for r in results:
        if r["tool_result_id"]:
            results_by_id.setdefault(r["tool_result_id"], []).append(r)

    history_uses_by_id = c.execute(
        f"""
        SELECT * FROM tool_events
        WHERE {where}
          AND event_type = 'tool_use_history'
        """,
        (session_id,),
    ).fetchall()
    history_by_id: Dict[str, List[sqlite3.Row]] = {}
    for r in history_uses_by_id:
        if r["tool_use_id"]:
            history_by_id.setdefault(r["tool_use_id"], []).append(r)

    # Wipe and rebuild for this session.
    if session_id is None:
        c.execute("DELETE FROM agent_calls WHERE session_id IS NULL")
    else:
        c.execute("DELETE FROM agent_calls WHERE session_id = ?", (session_id,))

    rows: List[Tuple] = []
    for s in starts:
        tool_use_id = s["tool_use_id"]
        start_ms = s["event_time_ms"]
        candidate_results = results_by_id.get(tool_use_id, [])
        result_row = candidate_results[0] if candidate_results else None
        result_ms = result_row["event_time_ms"] if result_row else None

        history_events = history_by_id.get(tool_use_id, []) if tool_use_id else []
        history_trace_ids = sorted({r["trace_id"] for r in history_events})

        # Child requests = subagent calls that happened between start and result
        child_request_ids: List[str] = []
        if start_ms is not None and result_ms is not None:
            for req in requests:
                if req["trace_id"] == s["trace_id"]:
                    continue
                rs = req["started_ms"]
                if rs is None or rs < start_ms or rs > result_ms:
                    continue
                if req["role_kind"] != "subagent":
                    continue
                child_request_ids.append(req["trace_id"])

        agent_call_id = stable_hash({"agent_call": s["trace_id"], "tool_use_id": tool_use_id})
        rows.append(
            (
                agent_call_id,
                s["trace_id"],
                session_id,
                tool_use_id,
                s["tool_name"],
                s["agent_label"],
                s["event_time"],
                start_ms,
                result_row["event_time"] if result_row else None,
                result_ms,
                (result_ms - start_ms) if (start_ms is not None and result_ms is not None) else None,
                "completed" if result_row else "open",
                s["input_preview"],
                result_row["input_preview"] if result_row else None,
                result_row["trace_id"] if result_row else None,
                json.dumps(child_request_ids),
                json.dumps(history_trace_ids),
            )
        )

    if rows:
        c.executemany(
            """
            INSERT INTO agent_calls
            (agent_call_id, parent_trace_id, session_id, tool_use_id, tool_name, agent_label,
             started_at, started_ms, completed_at, completed_ms, duration_ms, status,
             input_preview, result_preview, result_trace_id,
             child_request_ids_json, history_request_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    # Also update requests.parent_agent_call_id for child subagent calls.
    if rows:
        c.execute("UPDATE requests SET parent_agent_call_id = NULL WHERE session_id IS ? OR session_id = ?", (session_id, session_id))
        for row in rows:
            agent_call_id = row[0]
            child_ids = json.loads(row[15])
            for child in child_ids:
                c.execute(
                    "UPDATE requests SET parent_agent_call_id = ? WHERE trace_id = ?",
                    (agent_call_id, child),
                )


# ---------------------------------------------------------------------------
# Read API
# ---------------------------------------------------------------------------

def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    out: Dict[str, Any] = {}
    for key in row.keys():
        out[key] = row[key]
    return out


def _decode_jsons(data: Dict[str, Any], keys: Iterable[str]) -> None:
    for k in keys:
        if k in data and data[k]:
            try:
                data[k.replace("_json", "")] = json.loads(data[k])
            except Exception:
                data[k.replace("_json", "")] = data[k]


def list_sessions(include_archived: bool = False) -> List[Dict[str, Any]]:
    where = "" if include_archived else "WHERE s.archived = 0"
    result: List[Dict[str, Any]] = []
    for c in _all_conns():
        rows = c.execute(
            f"""
            SELECT s.*,
                   COUNT(r.trace_id) AS request_count_actual,
                   SUM(CASE WHEN r.role_kind = 'main' THEN 1 ELSE 0 END) AS main_requests,
                   SUM(CASE WHEN r.role_kind = 'subagent' THEN 1 ELSE 0 END) AS subagent_requests,
                   SUM(CASE WHEN r.role_kind = 'titler' THEN 1 ELSE 0 END) AS titler_requests,
                   SUM(CASE WHEN r.api = 'count_tokens' THEN 1 ELSE 0 END) AS token_requests,
                   MIN(r.model_mapped) AS sample_model_mapped
            FROM sessions s
            LEFT JOIN requests r ON r.session_id = s.session_id
            {where}
            GROUP BY s.session_id
            ORDER BY s.last_seen DESC
            """
        ).fetchall()
        result.extend(_row_to_dict(r) for r in rows)
    # 跨 db 统一按 last_seen 排序
    result.sort(key=lambda x: x.get("last_seen") or "", reverse=True)
    return result


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    c = _conn_for_session(session_id)
    s = c.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if not s:
        return None
    requests = list_requests({"session_id": session_id})
    agent_calls = c.execute(
        "SELECT * FROM agent_calls WHERE session_id = ? ORDER BY started_ms ASC",
        (session_id,),
    ).fetchall()
    sess = _row_to_dict(s) or {}
    sess["requests"] = requests
    sess["agent_calls"] = [_decode_agent_call(_row_to_dict(r)) for r in agent_calls]
    sess["title"] = next(
        (req.get("title") for req in requests if req.get("title")),
        None,
    )
    return sess


def _decode_agent_call(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not d:
        return d
    for k in ("child_request_ids_json", "history_request_ids_json"):
        if k in d and d[k]:
            try:
                d[k.replace("_json", "")] = json.loads(d[k])
            except Exception:
                d[k.replace("_json", "")] = []
    return d


def list_requests(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    filters = filters or {}
    session_id = filters.get("session_id")
    if session_id:
        conns = [_conn_for_session(session_id)]
    else:
        conns = _all_conns()
    where: List[str] = []
    params: List[Any] = []
    if "session_id" in filters and filters["session_id"]:
        where.append("session_id = ?")
        params.append(filters["session_id"])
    if "role_kind" in filters and filters["role_kind"]:
        where.append("role_kind = ?")
        params.append(filters["role_kind"])
    if "api" in filters and filters["api"]:
        where.append("api = ?")
        params.append(filters["api"])
    sql = """
        SELECT trace_id, session_id, api, role_kind, agent_label, model_requested, model_mapped,
               started_at, completed_at, started_ms, completed_ms, duration_ms,
               status, status_code, message_count, tool_count, advertises_agent_tool,
               history_hash, history_hash_approx, parent_trace_id, parent_match_kind,
               parent_added_steps, parent_agent_call_id, response_stop_reason, title,
               cc_version, cch
        FROM requests
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_ms ASC, trace_id ASC"
    out = []
    for c in conns:
        rows = c.execute(sql, params).fetchall()
        for r in rows:
            d = _row_to_dict(r)
            if d:
                # quick "tools used in response" preview
                uses = c.execute(
                    "SELECT tool_name FROM tool_events WHERE trace_id = ? AND event_type = 'tool_use_response' ORDER BY block_index ASC",
                    (r["trace_id"],),
                ).fetchall()
                d["response_tool_names"] = [u["tool_name"] for u in uses if u["tool_name"]]
                out.append(d)
    return out


def get_request(trace_id: str) -> Optional[Dict[str, Any]]:
    c = None
    row = None
    for conn in _all_conns():
        r = conn.execute("SELECT * FROM requests WHERE trace_id = ?", (trace_id,)).fetchone()
        if r:
            c = conn
            row = r
            break
    if not row:
        return None
    d = _row_to_dict(row) or {}
    _decode_jsons(
        d,
        [
            "headers_json",
            "tool_names_json",
            "prefix_hashes_json",
            "request_body_json",
            "converted_request_json",
            "response_body_json",
            "error_json",
            "extra_json",
            "response_tool_uses_json",
            "response_usage_json",
        ],
    )

    # Messages
    msgs = c.execute(
        "SELECT * FROM messages WHERE trace_id = ? ORDER BY msg_index ASC",
        (trace_id,),
    ).fetchall()
    d["messages"] = [_decode_message(_row_to_dict(m)) for m in msgs]

    # Tool events
    events = c.execute(
        "SELECT * FROM tool_events WHERE trace_id = ? ORDER BY event_time_ms ASC, block_index ASC",
        (trace_id,),
    ).fetchall()
    d["tool_events"] = [_decode_tool_event(_row_to_dict(e)) for e in events]

    # Parent + children
    children = c.execute(
        "SELECT trace_id, started_at, role_kind, message_count, status, title FROM requests WHERE parent_trace_id = ? ORDER BY started_ms ASC",
        (trace_id,),
    ).fetchall()
    d["children"] = [_row_to_dict(r) for r in children]

    if d.get("parent_trace_id"):
        parent = c.execute(
            "SELECT trace_id, started_at, role_kind, message_count, status, title FROM requests WHERE trace_id = ?",
            (d["parent_trace_id"],),
        ).fetchone()
        d["parent"] = _row_to_dict(parent)
    else:
        d["parent"] = None

    if d.get("parent_agent_call_id"):
        parent_call = c.execute(
            "SELECT * FROM agent_calls WHERE agent_call_id = ?",
            (d["parent_agent_call_id"],),
        ).fetchone()
        d["parent_agent_call"] = _decode_agent_call(_row_to_dict(parent_call))
    else:
        d["parent_agent_call"] = None

    # Agent calls launched FROM this request
    launched = c.execute(
        "SELECT * FROM agent_calls WHERE parent_trace_id = ? ORDER BY started_ms ASC",
        (trace_id,),
    ).fetchall()
    d["launched_agent_calls"] = [_decode_agent_call(_row_to_dict(r)) for r in launched]

    return d


def _decode_message(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not d:
        return d
    if d.get("full_content_json"):
        try:
            d["full_content"] = json.loads(d["full_content_json"])
        except Exception:
            d["full_content"] = None
    if d.get("block_types_json"):
        try:
            d["block_types"] = json.loads(d["block_types_json"])
        except Exception:
            pass
    return d


def _decode_tool_event(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not d:
        return d
    if d.get("raw_block_json"):
        try:
            d["raw_block"] = json.loads(d["raw_block_json"])
        except Exception:
            pass
    return d


# ---------------------------------------------------------------------------
# Higher-level analyses
# ---------------------------------------------------------------------------

def build_history_chains(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return a forest of LLM requests linked by parent_trace_id."""
    c = _conn_for_session(session_id)
    where = "WHERE session_id = ?" if session_id else ""
    params = (session_id,) if session_id else ()
    rows = c.execute(
        f"""
        SELECT trace_id, parent_trace_id, session_id, started_at, started_ms, role_kind,
               agent_label, model_requested, model_mapped, message_count, response_stop_reason,
               status, parent_match_kind, parent_added_steps, title, parent_agent_call_id
        FROM requests
        {where}
        ORDER BY started_ms ASC
        """,
        params,
    ).fetchall()

    nodes: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        d = _row_to_dict(r) or {}
        d["children"] = []
        nodes[d["trace_id"]] = d

    roots: List[Dict[str, Any]] = []
    for d in nodes.values():
        parent = d.get("parent_trace_id")
        if parent and parent in nodes:
            nodes[parent]["children"].append(d)
        else:
            roots.append(d)

    def _annotate_subtree(node: Dict[str, Any]) -> None:
        node["request_count"] = 1 + sum(_annotate_subtree(child) or child["request_count"] for child in node["children"])

    for root in roots:
        _annotate_subtree(root)
    return roots


def build_time_trajectory(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Group timeline events into classified time-segments (bursts).

    Algorithm:
      1. Pull events via build_timeline (sorted by event_time_ms).
      2. Compute inter-event gaps; split into bursts wherever
         gap > max(2000ms, 5 × median_gap).
      3. Classify each burst by role + activity composition.

    Output:
      {
        "events":   [<event_obj>, ...],
        "segments": [{
            "id", "start_ms", "end_ms", "duration_ms",
            "classification",   # canonical key
            "label",            # human-readable
            "event_start_index","event_end_index","event_count",
            "stats": { llm_starts, llm_ends, tool_uses, tool_results, unique_agents }
        }, ...]
      }
    """
    events = build_timeline(session_id)
    if not events:
        return {"events": [], "segments": []}

    times = [e.get("event_time_ms") or 0 for e in events]

    # Gap-based segmentation
    if len(events) == 1:
        breakpoints = [0, 1]
    else:
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2] if sorted_gaps else 0
        threshold = max(2000, median_gap * 5)
        breakpoints = [0]
        for i, g in enumerate(gaps):
            if g > threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(events))

    segments: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(zip(breakpoints[:-1], breakpoints[1:])):
        seg = _classify_burst(events, start, end)
        seg["id"] = f"seg-{idx}"
        segments.append(seg)

    return {"events": events, "segments": segments}


def _classify_burst(events: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Dict[str, Any]:
    sub = events[start_idx:end_idx]
    n_llm_start = sum(1 for e in sub if e.get("kind") == "llm_request_start")
    n_llm_end = sum(1 for e in sub if e.get("kind") == "llm_request_end")
    n_tool_use = sum(1 for e in sub if e.get("kind") == "tool_use_response")
    n_tool_result = sum(1 for e in sub if e.get("kind") == "tool_result")

    role_kinds: set = set()
    agent_labels: set = set()
    for e in sub:
        r = e.get("role_kind")
        if r:
            role_kinds.add(r)
        else:
            # tool events use agent_label as proxy
            label = e.get("agent_label") or "main"
            role_kinds.add("main" if label == "main" else "subagent")
        l = e.get("agent_label")
        if l and l not in ("main", "titler"):
            agent_labels.add(l)

    start_ms = sub[0].get("event_time_ms") or 0
    end_ms = sub[-1].get("event_time_ms") or 0

    has_main = "main" in role_kinds
    has_subagent = "subagent" in role_kinds
    has_titler = "titler" in role_kinds
    has_tools = n_tool_use > 0

    sorted_labels = sorted(agent_labels)

    if has_titler and not has_main and not has_subagent:
        classification = "titler"
        label = f"Titler · {n_llm_start} call"
    elif has_subagent and has_main:
        labels_str = ", ".join(sorted_labels[:2]) if sorted_labels else "subagents"
        classification = "agent_dispatch"
        label = f"Agent dispatch · {labels_str}" + ("…" if len(sorted_labels) > 2 else "")
    elif has_subagent and not has_main:
        labels_str = ", ".join(sorted_labels[:2]) if sorted_labels else "subagent"
        classification = "subagent"
        label = f"Subagent · {labels_str}" + ("…" if len(sorted_labels) > 2 else "")
    elif has_main and has_tools:
        classification = "main_tools"
        plural = "s" if n_tool_use != 1 else ""
        label = f"Main + tools · {n_tool_use} tool{plural}"
    elif has_main:
        classification = "main_thinking"
        plural = "s" if n_llm_start != 1 else ""
        label = f"Main thinking · {n_llm_start} call{plural}"
    else:
        classification = "misc"
        label = f"{len(sub)} events"

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "classification": classification,
        "label": label,
        "event_start_index": start_idx,
        "event_end_index": end_idx,
        "event_count": len(sub),
        "stats": {
            "llm_starts": n_llm_start,
            "llm_ends": n_llm_end,
            "tool_uses": n_tool_use,
            "tool_results": n_tool_result,
            "unique_agents": sorted_labels,
        },
    }


def build_prefix_trie(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build a compressed trie of all requests' prefix_hashes.

    Each output node represents either a branching point or a terminal where
    one (or more) requests end. Chains with a single child and no terminals
    are compressed into the next node's `step_count`.

    Node shape:
      {
        "node_id":          stable id ("branch:<hash>" or "req:<trace_id>"),
        "step_count":       int,    # how many message-steps this edge spans
        "first_step_hash":  str | None,
        "last_step_hash":   str | None,
        "terminal_requests": [ {trace_id, role_kind, ...}, ... ],
        "children":         [ <node>, ... ]
      }
    """
    c = _conn_for_session(session_id)
    where = "WHERE session_id = ?" if session_id else ""
    params = (session_id,) if session_id else ()
    rows = c.execute(
        f"""
        SELECT trace_id, role_kind, agent_label, model_mapped, model_requested,
               title, message_count, prefix_hashes_json, started_ms, status,
               response_stop_reason, duration_ms
        FROM requests
        {where}
        ORDER BY started_ms ASC
        """,
        params,
    ).fetchall()

    # Raw trie: each node is { 'children': {hash: child_node}, 'requests': [...], 'step': hash }
    raw_root: Dict[str, Any] = {"children": {}, "requests": []}

    for r in rows:
        hashes_json = r["prefix_hashes_json"]
        if not hashes_json:
            continue
        try:
            hashes = json.loads(hashes_json)
        except Exception:
            continue
        if not hashes:
            continue
        req_summary = {
            "trace_id": r["trace_id"],
            "role_kind": r["role_kind"],
            "agent_label": r["agent_label"],
            "model_mapped": r["model_mapped"],
            "model_requested": r["model_requested"],
            "title": r["title"],
            "message_count": r["message_count"],
            "status": r["status"],
            "response_stop_reason": r["response_stop_reason"],
            "started_ms": r["started_ms"],
            "duration_ms": r["duration_ms"],
            "depth": len(hashes),
        }
        cur = raw_root
        for h in hashes:
            child = cur["children"].get(h)
            if child is None:
                child = {"children": {}, "requests": [], "step": h}
                cur["children"][h] = child
            cur = child
        cur["requests"].append(req_summary)

    # Compress chains during serialization.
    # edge_steps = list of hashes from the previous branching point down to (and including) this node.
    def _build(node: Dict[str, Any], edge_steps: List[str]) -> Dict[str, Any]:
        # Compress: if exactly one child and no terminal requests, keep going.
        while len(node["children"]) == 1 and not node["requests"]:
            ((h, child),) = node["children"].items()
            edge_steps.append(h)
            node = child
        terminals = node["requests"]
        children_items = list(node["children"].items())
        # Stable node_id: terminal → first req's trace_id; otherwise the last hash on this edge.
        if terminals:
            node_id = f"req:{terminals[0]['trace_id']}"
        elif edge_steps:
            node_id = f"branch:{edge_steps[-1]}"
        else:
            node_id = "branch:root"
        out = {
            "node_id": node_id,
            "step_count": len(edge_steps),
            "first_step_hash": edge_steps[0] if edge_steps else None,
            "last_step_hash": edge_steps[-1] if edge_steps else None,
            "terminal_requests": terminals,
            "children": [_build(ch, [k]) for k, ch in children_items],
        }
        # Aggregate metrics for quick display
        def _count(n: Dict[str, Any]) -> int:
            return len(n["terminal_requests"]) + sum(_count(c) for c in n["children"])
        out["subtree_request_count"] = _count(out)
        return out

    roots = [_build(ch, [k]) for k, ch in raw_root["children"].items()]
    return roots


def build_agent_tree(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return a tree of agent activity for the session.

    Top-level nodes: each "main-agent conversation" (a chain of main requests
    linked by parent_trace_id) and any free-standing titler requests.
    Inside each main request: agent_call children for every Agent/Task tool_use.
    Inside each agent_call: subagent request chains that ran during the call.
    """
    c = _conn_for_session(session_id)
    where = "WHERE session_id = ?" if session_id else ""
    params = (session_id,) if session_id else ()

    requests = list(c.execute(
        f"""
        SELECT * FROM requests
        {where}
        ORDER BY started_ms ASC
        """,
        params,
    ).fetchall())
    requests_by_id: Dict[str, Dict[str, Any]] = {r["trace_id"]: _row_to_dict(r) for r in requests}

    agent_calls = c.execute(
        f"""
        SELECT * FROM agent_calls
        {where}
        ORDER BY started_ms ASC
        """,
        params,
    ).fetchall()
    calls_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for ac in agent_calls:
        d = _decode_agent_call(_row_to_dict(ac)) or {}
        calls_by_parent.setdefault(d["parent_trace_id"], []).append(d)

    children_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for r in requests_by_id.values():
        if r.get("parent_trace_id") and r["parent_trace_id"] in requests_by_id:
            children_by_parent.setdefault(r["parent_trace_id"], []).append(r)

    def render(request: Dict[str, Any]) -> Dict[str, Any]:
        node = {
            "kind": "request",
            "trace_id": request["trace_id"],
            "title": request.get("title"),
            "role_kind": request["role_kind"],
            "agent_label": request["agent_label"],
            "started_at": request["started_at"],
            "model_requested": request["model_requested"],
            "model_mapped": request["model_mapped"],
            "status": request["status"],
            "message_count": request["message_count"],
            "duration_ms": request.get("duration_ms"),
            "stop_reason": request.get("response_stop_reason"),
            "children": [],
        }
        # 1) Agent calls launched FROM this request
        for call in calls_by_parent.get(request["trace_id"], []):
            call_node = {
                "kind": "agent_call",
                "agent_call_id": call["agent_call_id"],
                "agent_label": call.get("agent_label"),
                "tool_name": call.get("tool_name"),
                "started_at": call.get("started_at"),
                "completed_at": call.get("completed_at"),
                "status": call.get("status"),
                "input_preview": call.get("input_preview"),
                "result_preview": call.get("result_preview"),
                "duration_ms": call.get("duration_ms"),
                "child_request_ids": call.get("child_request_ids") or [],
                "children": [],
            }
            seen_children: set = set()
            for cid in call.get("child_request_ids") or []:
                if cid in seen_children:
                    continue
                seen_children.add(cid)
                child = requests_by_id.get(cid)
                if child:
                    call_node["children"].append(render(child))
            node["children"].append(call_node)
        # 2) Continuation main requests linked by parent_trace_id (same agent next turn)
        for child in children_by_parent.get(request["trace_id"], []):
            node["children"].append(render(child))
        return node

    roots: List[Dict[str, Any]] = []
    for r in requests:
        rd = requests_by_id[r["trace_id"]]
        # Roots: anything without a request-level parent AND without an agent_call parent.
        if rd.get("parent_trace_id") in requests_by_id:
            continue
        if rd.get("parent_agent_call_id"):
            continue
        roots.append(render(rd))

    return roots


def build_timeline(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    c = _conn_for_session(session_id)
    where = "WHERE session_id = ?" if session_id else ""
    params = (session_id,) if session_id else ()

    requests = c.execute(
        f"""
        SELECT trace_id, session_id, role_kind, agent_label, model_requested, model_mapped,
               started_at, started_ms, completed_at, completed_ms, status, message_count,
               response_stop_reason, parent_agent_call_id, title
        FROM requests
        {where}
        ORDER BY started_ms ASC
        """,
        params,
    ).fetchall()

    events: List[Dict[str, Any]] = []
    for r in requests:
        events.append(
            {
                "kind": "llm_request_start",
                "trace_id": r["trace_id"],
                "session_id": r["session_id"],
                "role_kind": r["role_kind"],
                "agent_label": r["agent_label"],
                "model_requested": r["model_requested"],
                "model_mapped": r["model_mapped"],
                "title": r["title"],
                "message_count": r["message_count"],
                "event_time": r["started_at"],
                "event_time_ms": r["started_ms"],
                "parent_agent_call_id": r["parent_agent_call_id"],
            }
        )
        if r["completed_at"]:
            events.append(
                {
                    "kind": "llm_request_end",
                    "trace_id": r["trace_id"],
                    "session_id": r["session_id"],
                    "role_kind": r["role_kind"],
                    "agent_label": r["agent_label"],
                    "model_requested": r["model_requested"],
                    "model_mapped": r["model_mapped"],
                    "title": r["title"],
                    "status": r["status"],
                    "stop_reason": r["response_stop_reason"],
                    "event_time": r["completed_at"],
                    "event_time_ms": r["completed_ms"],
                    "parent_agent_call_id": r["parent_agent_call_id"],
                }
            )

    tool_events = c.execute(
        f"""
        SELECT te.* FROM tool_events te
        {('JOIN requests r ON r.trace_id = te.trace_id WHERE r.session_id = ?' if session_id else '')}
        AND event_type IN ('tool_use_response','tool_result')
        ORDER BY event_time_ms ASC
        """ if session_id else
        """
        SELECT * FROM tool_events
        WHERE event_type IN ('tool_use_response','tool_result')
        ORDER BY event_time_ms ASC
        """,
        params if session_id else (),
    ).fetchall()
    for e in tool_events:
        events.append(
            {
                "kind": e["event_type"],
                "trace_id": e["trace_id"],
                "session_id": e["session_id"],
                "agent_label": e["agent_label"] or "main",
                "tool_name": e["tool_name"],
                "tool_use_id": e["tool_use_id"],
                "tool_result_id": e["tool_result_id"],
                "input_preview": e["input_preview"],
                "event_time": e["event_time"],
                "event_time_ms": e["event_time_ms"],
                "event_id": e["event_id"],
            }
        )

    events.sort(key=lambda x: (x.get("event_time_ms") or 0, x.get("event_time") or "", x.get("trace_id") or ""))
    for i, e in enumerate(events, start=1):
        e["index"] = i
    return events


def get_tool_event(event_id: str) -> Optional[Dict[str, Any]]:
    c = None
    row = None
    for conn in _all_conns():
        r = conn.execute("SELECT * FROM tool_events WHERE event_id = ?", (event_id,)).fetchone()
        if r:
            c = conn
            row = r
            break
    if not row:
        return None
    d = _decode_tool_event(_row_to_dict(row)) or {}
    if d.get("trace_id"):
        # also include matching pair
        if d["event_type"] == "tool_use_response" and d.get("tool_use_id"):
            partner = c.execute(
                "SELECT * FROM tool_events WHERE tool_result_id = ? ORDER BY event_time_ms ASC LIMIT 1",
                (d["tool_use_id"],),
            ).fetchone()
            d["partner"] = _decode_tool_event(_row_to_dict(partner))
        elif d["event_type"] == "tool_result" and d.get("tool_result_id"):
            partner = c.execute(
                "SELECT * FROM tool_events WHERE tool_use_id = ? AND event_type = 'tool_use_response' ORDER BY event_time_ms ASC LIMIT 1",
                (d["tool_result_id"],),
            ).fetchone()
            d["partner"] = _decode_tool_event(_row_to_dict(partner))
    return d


def get_agent_call(agent_call_id: str) -> Optional[Dict[str, Any]]:
    c = None
    row = None
    for conn in _all_conns():
        r = conn.execute("SELECT * FROM agent_calls WHERE agent_call_id = ?", (agent_call_id,)).fetchone()
        if r:
            c = conn
            row = r
            break
    if not row:
        return None
    d = _decode_agent_call(_row_to_dict(row)) or {}
    # attach parent + child request summaries
    if d.get("parent_trace_id"):
        parent = c.execute(
            "SELECT trace_id, role_kind, agent_label, started_at, status, message_count, title FROM requests WHERE trace_id = ?",
            (d["parent_trace_id"],),
        ).fetchone()
        d["parent_request"] = _row_to_dict(parent)
    children = []
    for cid in d.get("child_request_ids") or []:
        child = c.execute(
            "SELECT trace_id, role_kind, agent_label, started_at, status, message_count, title, response_stop_reason FROM requests WHERE trace_id = ?",
            (cid,),
        ).fetchone()
        if child:
            children.append(_row_to_dict(child))
    d["child_requests"] = children
    return d


def snapshot_stats() -> Dict[str, Any]:
    all_conns = _all_conns()

    def total(sql: str) -> int:
        return sum(c.execute(sql).fetchone()[0] for c in all_conns)

    stats: Dict[str, Any] = {
        "trace_enabled": is_trace_enabled(),
        "trace_db": str(DB_PATH),
        "op_db_count": max(0, len(all_conns) - 1),  # 算子 db 数量（除默认 trace.db）
        "include_system_in_prefix": TRACE_INCLUDE_SYSTEM,
        "generated_at": utc_now_iso(),
    }
    stats["session_count"] = total("SELECT COUNT(*) FROM sessions")
    stats["request_count"] = total("SELECT COUNT(*) FROM requests")
    stats["message_request_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE api='messages'"
    )
    stats["count_tokens_request_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE api='count_tokens'"
    )
    stats["main_request_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE role_kind='main'"
    )
    stats["subagent_request_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE role_kind='subagent'"
    )
    stats["titler_request_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE role_kind='titler'"
    )
    stats["external_request_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE role_kind='external'"
    )
    stats["agent_call_count"] = total("SELECT COUNT(*) FROM agent_calls")
    stats["open_agent_call_count"] = total(
        "SELECT COUNT(*) FROM agent_calls WHERE status != 'completed'"
    )
    stats["tool_event_count"] = total(
        "SELECT COUNT(*) FROM tool_events WHERE event_type IN ('tool_use_response','tool_result')"
    )
    stats["history_link_count"] = total(
        "SELECT COUNT(*) FROM requests WHERE parent_trace_id IS NOT NULL"
    )
    return stats


# ---------------------------------------------------------------------------
# Compatibility shims used by older callers
# ---------------------------------------------------------------------------

def headers_to_trace(items: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    return headers_to_dict(items)


def trace_request_started(**kwargs: Any) -> None:
    record_request_started(**kwargs)


def trace_request_completed(**kwargs: Any) -> None:
    record_request_completed(**kwargs)


def trace_request_failed(**kwargs: Any) -> None:
    record_request_failed(**kwargs)


def clear_trace_events() -> None:
    clear_traces()
