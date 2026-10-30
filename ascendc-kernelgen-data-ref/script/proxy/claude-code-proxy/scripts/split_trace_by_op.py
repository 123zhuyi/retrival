#!/usr/bin/env python3
"""按算子拆分 trace.db。

读 cc_traces/trace.db，解析每个 session 第一个请求的 prompt 拿算子信息
（从"算子描述文件为 <path>"提取 basename），按 op_key 把该 session 的所有数据
（sessions / requests / messages / tool_events / agent_calls）复制到独立 {op_key}.db。

默认拆分后从源 trace.db 删除已拆分的 session（避免重复拆分），用 --keep-source 保留。

用法：
    python3 scripts/split_trace_by_op.py
    python3 scripts/split_trace_by_op.py --trace-db cc_traces/trace.db --out-dir cc_traces
    python3 scripts/split_trace_by_op.py --keep-source
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional


# 批跑脚本的 PROMPT 里有"算子描述文件为 <path>.py"，正则提取 path
OP_FILE_PATTERN = re.compile(r"算子描述文件为\s+(\S+\.py)")

# trace.db 的 schema（跟 trace_db.py 一致，复制过来避免 import 依赖）
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL DEFAULT 'header',
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
    role_kind TEXT NOT NULL,
    agent_label TEXT NOT NULL,
    model_requested TEXT,
    model_mapped TEXT,
    started_at TEXT NOT NULL,
    started_ms INTEGER,
    completed_at TEXT,
    completed_ms INTEGER,
    duration_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'inflight',
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
    role_or_source TEXT,
    content_kind TEXT,
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
    location TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message_index INTEGER,
    block_index INTEGER,
    role TEXT,
    tool_name TEXT,
    tool_use_id TEXT,
    tool_result_id TEXT,
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
    agent_label TEXT,
    started_at TEXT,
    started_ms INTEGER,
    completed_at TEXT,
    completed_ms INTEGER,
    duration_ms INTEGER,
    status TEXT,
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


def extract_op_key(request_body_json: str) -> Optional[str]:
    """从请求 body 解析算子描述文件路径，返回 op_key（basename 去后缀）。

    遍历 messages，找 user 消息里的"算子描述文件为 <path>.py"。
    """
    try:
        body = json.loads(request_body_json)
    except Exception:
        return None
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
        for m in OP_FILE_PATTERN.finditer(text):
            path = m.group(1).strip().rstrip(",，")
            if "/path/to/" in path:
                continue
            try:
                return Path(path).stem
            except Exception:
                # 兜底：手动取 basename 去后缀
                name = path.rsplit("/", 1)[-1]
                return name.rsplit(".", 1)[0] if "." in name else name
    return None


def get_session_op_map(conn: sqlite3.Connection) -> Dict[str, str]:
    """返回 session_id → op_key 映射（每个 session 第一个请求解析）。"""
    rows = conn.execute(
        "SELECT session_id, request_body_json FROM requests "
        "WHERE session_id IS NOT NULL ORDER BY started_ms ASC"
    ).fetchall()

    session_op: Dict[str, str] = {}
    for row in rows:
        session_id = row["session_id"]
        if session_id in session_op:
            continue
        body_json = row["request_body_json"]
        if not body_json:
            continue
        op_key = extract_op_key(body_json)
        if op_key:
            session_op[session_id] = op_key
    return session_op


def init_db(conn: sqlite3.Connection) -> None:
    """初始化目标 db 的 schema（幂等）。"""
    conn.executescript(SCHEMA)


def _copy_table(source: sqlite3.Connection, dest: sqlite3.Connection,
                table: str, where: str, params: tuple) -> int:
    """复制符合条件的行到目标 db，返回复制行数。"""
    rows = source.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchall()
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)
    count = 0
    for row in rows:
        try:
            dest.execute(
                f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})",
                tuple(row),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    return count


def copy_session(source: sqlite3.Connection, dest: sqlite3.Connection, session_id: str) -> int:
    """复制单个 session 的所有数据到 dest db。返回复制行数。"""
    count = 0
    count += _copy_table(source, dest, "sessions", "session_id = ?", (session_id,))
    count += _copy_table(source, dest, "requests", "session_id = ?", (session_id,))
    # messages 通过 trace_id 关联（没有 session_id 字段）
    count += _copy_table(
        source, dest, "messages",
        "trace_id IN (SELECT trace_id FROM requests WHERE session_id = ?)",
        (session_id,),
    )
    count += _copy_table(source, dest, "tool_events", "session_id = ?", (session_id,))
    count += _copy_table(source, dest, "agent_calls", "session_id = ?", (session_id,))
    return count


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    """从源 db 删除已拆分的 session 及其关联数据。"""
    conn.execute(
        "DELETE FROM messages WHERE trace_id IN "
        "(SELECT trace_id FROM requests WHERE session_id = ?)",
        (session_id,),
    )
    conn.execute("DELETE FROM tool_events WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM agent_calls WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM requests WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


def main() -> int:
    parser = argparse.ArgumentParser(description="按算子拆分 trace.db")
    default_db = os.path.join(os.environ.get("CC_TRACE_DIR", "cc_traces"), "trace.db")
    parser.add_argument("--trace-db", default=default_db, help=f"源 trace.db 路径（默认 {default_db}）")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认跟 trace-db 同目录）")
    parser.add_argument("--keep-source", action="store_true",
                        help="保留 trace.db 里已拆分的 session（默认删除）")
    args = parser.parse_args()

    trace_db = Path(args.trace_db)
    if not trace_db.exists():
        print(f"错误：找不到 trace.db: {trace_db}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else trace_db.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(str(trace_db))
    source.row_factory = sqlite3.Row

    session_op = get_session_op_map(source)
    print(f"识别到 {len(session_op)} 个算子 session")

    if not session_op:
        print("没有可拆分的算子 session，退出")
        source.close()
        return 0

    # 按 op_key 分组 session
    op_sessions: Dict[str, List[str]] = {}
    for session_id, op_key in session_op.items():
        op_sessions.setdefault(op_key, []).append(session_id)

    print(f"涉及 {len(op_sessions)} 个算子：{', '.join(sorted(op_sessions.keys()))}")

    total_copied = 0
    for op_key in sorted(op_sessions.keys()):
        session_ids = op_sessions[op_key]
        out_path = out_dir / f"{op_key}.db"
        print(f"  导出 {op_key}.db ({len(session_ids)} session) ... ", end="", flush=True)

        # 如果目标 db 已存在，追加（同算子多次 retry 或多次拆分会聚合）
        out_conn = sqlite3.connect(str(out_path))
        out_conn.row_factory = sqlite3.Row
        init_db(out_conn)

        op_count = 0
        for session_id in session_ids:
            op_count += copy_session(source, out_conn, session_id)

        out_conn.commit()
        out_conn.close()
        print(f"{op_count} 行")
        total_copied += op_count

    # 默认从源 db 删除已拆分的 session（避免重复拆分）
    if not args.keep_source:
        print("从源 trace.db 删除已拆分的 session ...")
        for session_id in session_op:
            delete_session(source, session_id)
        source.commit()
        print(f"  源 trace.db 已清理（保留非算子 session）")

    source.close()
    print(f"拆分完成：共复制 {total_copied} 行到 {len(op_sessions)} 个算子 db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
