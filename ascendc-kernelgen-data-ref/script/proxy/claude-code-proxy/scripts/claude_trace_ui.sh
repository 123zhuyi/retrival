#!/usr/bin/env bash
set -euo pipefail
WORKDIR=$(pwd)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${CC_TRACE_HOST:-127.0.0.1}"
PORT="${CC_TRACE_PORT:-8082}"
BASE_URL="${CC_TRACE_BASE_URL:-http://${HOST}:${PORT}}"
MODEL="${CC_TRACE_CLAUDE_MODEL:-claude-3-5-haiku-20241022}"
SERVER_LOG="${CC_TRACE_SERVER_LOG:-${ROOT_DIR}/cc_traces/gateway.log}"
SERVER_PID_FILE="${CC_TRACE_SERVER_PID_FILE:-${ROOT_DIR}/cc_traces/gateway.pid}"
CLEAR_TRACE=0
START_SERVER=1
CLAUDE_ARGS=()

usage() {
  cat <<'USAGE'
Usage: scripts/claude_trace_ui.sh [options] [-- extra claude args]

Starts the trace gateway if needed, then launches Claude Code pointed at it.

Options:
  --clear-trace          Clear /api/v2/traces before launching Claude Code.
  --no-server           Do not auto-start the FastAPI gateway.
  --port PORT           Gateway port, default 8082.
  --host HOST           Gateway host, default 127.0.0.1.
  --base-url URL        Full gateway URL, default http://HOST:PORT.
  --model MODEL         Claude model name sent by Claude Code, default claude-3-5-haiku-20241022.
  --help                Show this help.

Environment knobs:
  CC_TRACE_CLAUDE_MODEL     Same as --model.
  CC_TRACE_BASE_URL         Same as --base-url.
  CC_TRACE_PORT             Same as --port.
  CC_TRACE_HOST             Same as --host.
  CC_TRACE_PROVIDER         PREFERRED_PROVIDER for the gateway, e.g. openai.
  CC_TRACE_BIG_MODEL        BIG_MODEL for the gateway, e.g. qwen-plus.
  CC_TRACE_SMALL_MODEL      SMALL_MODEL for the gateway, e.g. qwen-turbo.
  OPENAI_BASE_URL           Upstream OpenAI-compatible/Qwen endpoint.
  OPENAI_API_KEY            Upstream API key.

Examples:
  OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  OPENAI_API_KEY=sk-... \
  CC_TRACE_BIG_MODEL=qwen-plus \
  CC_TRACE_SMALL_MODEL=qwen-turbo \
  scripts/claude_trace_ui.sh --clear-trace

  scripts/claude_trace_ui.sh -- --permission-mode bypassPermissions
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clear-trace)
      CLEAR_TRACE=1
      shift
      ;;
    --no-server)
      START_SERVER=0
      shift
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      BASE_URL="http://${HOST}:${PORT}"
      shift 2
      ;;
    --host)
      HOST="${2:?missing value for --host}"
      BASE_URL="http://${HOST}:${PORT}"
      shift 2
      ;;
    --base-url)
      BASE_URL="${2:?missing value for --base-url}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:?missing value for --workdir}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      CLAUDE_ARGS+=("$@")
      break
      ;;
    *)
      CLAUDE_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v claude >/dev/null 2>&1; then
  echo "error: claude command not found. Install Claude Code first: npm install -g @anthropic-ai/claude-code" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv command not found. Install uv or start the gateway yourself and rerun with --no-server." >&2
  exit 1
fi

mkdir -p "$(dirname "$SERVER_LOG")"

server_ready() {
  curl -fsS "${BASE_URL}/api/v2/stats" >/dev/null 2>&1
}

if [[ "$START_SERVER" == "1" ]]; then
  if server_ready; then
    echo "trace gateway already running at ${BASE_URL}"
  else
    echo "starting trace gateway at ${BASE_URL}"
    (
      export PORT="$PORT"
      export HOST="0.0.0.0"
      export WORKERS="${WORKERS:-1}"
      export PREFERRED_PROVIDER="${CC_TRACE_PROVIDER:-${PREFERRED_PROVIDER:-openai}}"
      export BIG_MODEL="${CC_TRACE_BIG_MODEL:-${BIG_MODEL:-gpt-4.1}}"
      export SMALL_MODEL="${CC_TRACE_SMALL_MODEL:-${SMALL_MODEL:-gpt-4.1-mini}}"
      exec uv run uvicorn server:app --host 0.0.0.0 --port "$PORT"
    ) >"$SERVER_LOG" 2>&1 &
    echo $! >"$SERVER_PID_FILE"

    for _ in $(seq 1 40); do
      if server_ready; then
        break
      fi
      sleep 0.25
    done

    if ! server_ready; then
      echo "error: trace gateway failed to start. Log: ${SERVER_LOG}" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      exit 1
    fi
  fi
fi

if [[ "$CLEAR_TRACE" == "1" ]]; then
  echo "clearing trace at ${BASE_URL}/api/v2/traces"
  curl -fsS -X DELETE "${BASE_URL}/api/v2/traces" >/dev/null
fi

echo "trace ui: ${BASE_URL}/trace"
echo "claude base url: ${BASE_URL}"
echo "claude model: ${MODEL}"
echo "gateway log: ${SERVER_LOG}"

cd $WORKDIR
# --setting-sources local prevents user/global Claude settings from overriding ANTHROPIC_BASE_URL.
exec env \
  -u ANTHROPIC_AUTH_TOKEN \
  -u ANTHROPIC_MODEL \
  -u ANTHROPIC_DEFAULT_OPUS_MODEL \
  -u ANTHROPIC_DEFAULT_SONNET_MODEL \
  -u ANTHROPIC_DEFAULT_HAIKU_MODEL \
  ANTHROPIC_BASE_URL="$BASE_URL" \
  ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-trace-key}" \
  DISABLE_TELEMETRY="${DISABLE_TELEMETRY:-1}" \
  claude --setting-sources local --model "$MODEL" "${CLAUDE_ARGS[@]}"
