#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# trace_proxy.sh — trace proxy 启停函数库（精简版）
#
# 被 run_benchmark_ascendc.sh source 使用，提供两个函数：
#   trace_proxy_start <port>    起 proxy（监听指定端口）+ 就绪探测
#   trace_proxy_stop            进程组 kill proxy
#
# 设计要点：
#   - 只管启停 proxy，不碰 settings.json，不碰 proxy/.env（都由用户自己配）
#   - 用 setsid 让 proxy 独立成进程组，stop 时进程组 kill 连带 uvicorn worker 一起清
#   - 去掉 --reload（并发场景热重载会丢请求）
#   - 函数幂等，stop 多次调用安全
# ----------------------------------------------------------------------------------------------------------

# 全局状态（start 写入，stop 读取）
PROXY_PID=""
PROXY_PGID=""
PROXY_LOG=""
PROXY_TRACE_DIR=""   # 本次批跑的 trace 目录（带时间戳，避免多次批跑覆盖）

# ------------------------------------------------------------------------------
# trace_proxy_start <port>
# 成功 return 0，失败 return 1（由调用方 exit，trap 会负责清理）
# 依赖主脚本先定义 SCRIPT_DIR（脚本所在目录）
# ------------------------------------------------------------------------------
trace_proxy_start() {
    local port="$1"
    local proxy_dir="${SCRIPT_DIR}/proxy/claude-code-proxy"

    echo "[trace] === trace_proxy_start: port=$port dir=$proxy_dir ==="

    # --- 1. 前置检查 ---
    if ! command -v uvicorn >/dev/null 2>&1; then
        echo "[trace] 错误：未找到 uvicorn 命令，请先安装 fastapi/uvicorn" >&2
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        echo "[trace] 错误：未找到 curl（就绪探测用）" >&2
        return 1
    fi
    if [[ ! -f "$proxy_dir/server.py" ]]; then
        echo "[trace] 错误：$proxy_dir/server.py 不存在" >&2
        return 1
    fi
    if [[ ! -f "$proxy_dir/.env" ]]; then
        echo "[trace] 错误：$proxy_dir/.env 不存在（请参考 .env.example 自己配）" >&2
        return 1
    fi
    if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$port "; then
        echo "[trace] 错误：端口 $port 已被占用" >&2
        return 1
    fi

    # --- 2. 生成唯一的 trace 目录（每次批跑独立子目录，避免覆盖）---
    local timestamp=$(date +%Y%m%d_%H%M%S)
    PROXY_TRACE_DIR="${proxy_dir}/cc_traces/batch_${timestamp}"
    mkdir -p "$PROXY_TRACE_DIR"
    export CC_TRACE_DIR="$PROXY_TRACE_DIR"
    echo "[trace] trace 目录: $PROXY_TRACE_DIR"

    # --- 3. 后台启动 proxy（去 --reload；setsid 独立进程组；继承 CC_TRACE_DIR）---
    PROXY_LOG="/tmp/cc-proxy.$$.log"
    local orig_dir="$(pwd)"
    cd "$proxy_dir"
    setsid nohup uvicorn server:app \
        --host 0.0.0.0 --port "$port" \
        > "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!
    disown "$PROXY_PID" 2>/dev/null || true   # 让 bash 放弃跟踪 proxy，避免无参数 wait 卡住
    # setsid 让 proxy 成为新会话 leader，PGID == PID；ps 兜底确认
    PROXY_PGID=$(ps -o pgid= -p "$PROXY_PID" 2>/dev/null | tr -d ' ')
    [[ -z "$PROXY_PGID" ]] && PROXY_PGID="$PROXY_PID"
    cd "$orig_dir"
    echo "[trace] proxy 后台启动 (pid=$PROXY_PID, pgid=$PROXY_PGID)"

    # --- 3. 就绪探测（轮询 15s）---
    local i
    for i in $(seq 1 30); do
        if ! kill -0 "$PROXY_PID" 2>/dev/null; then
            echo "[trace] 错误：proxy 进程已退出，查看日志：$PROXY_LOG" >&2
            tail -20 "$PROXY_LOG" >&2 2>/dev/null || true
            return 1
        fi
        if curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$port/" 2>/dev/null; then
            echo "[trace] proxy ready (port=$port, log=$PROXY_LOG)"
            echo "[trace] === trace_proxy_start 完成 ==="
            return 0
        fi
        sleep 0.5
    done
    echo "[trace] 错误：proxy 就绪探测超时（15s），查看日志：$PROXY_LOG" >&2
    tail -20 "$PROXY_LOG" >&2 2>/dev/null || true
    return 1
}

# ------------------------------------------------------------------------------
# trace_proxy_stop
# 幂等：变量为空或进程不存在则跳过，可被 trap 多次安全调用
# ------------------------------------------------------------------------------
trace_proxy_stop() {
    # 进程组 kill proxy（连带 uvicorn worker）
    if [[ -n "$PROXY_PGID" ]]; then
        kill -- -"$PROXY_PGID" 2>/dev/null \
            || kill "$PROXY_PID" 2>/dev/null \
            || true
        echo "[trace] proxy 已停止 (pid=$PROXY_PID, pgid=$PROXY_PGID)"
        PROXY_PID=""
        PROXY_PGID=""
    fi

    echo "[trace] === trace_proxy_stop 完成 ==="
}
