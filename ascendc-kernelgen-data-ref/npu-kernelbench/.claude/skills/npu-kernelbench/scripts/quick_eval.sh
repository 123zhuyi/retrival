#!/usr/bin/env bash
# Quick evaluation helper for a single npu-kernelbench solution.
#
# Usage:
#     ./quick_eval.sh <problem_dir> <solution.json> [output.jsonl]
#
# Environment variables:
#     NPU_KERNELBENCH_RUNNER   "persistent" (default) or "isolated"
#     NPU_KERNELBENCH_NPU_IDS  comma-separated NPU IDs (default: 0)
#     NPU_KERNELBENCH_MODE     "correctness_only" or empty for full benchmark
#     NPU_KERNELBENCH_LIMIT    max workloads to evaluate
#     NPU_KERNELBENCH_VERBOSE  set to any value to enable --verbose

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [ $# -lt 2 ]; then
    echo "Usage: $(basename "$0") <problem_dir> <solution.json> [output.jsonl]"
    exit 1
fi

PROBLEM_DIR="$1"
SOLUTION_JSON="$2"
OUTPUT_JSONL="${3:-/tmp/npu-kernelbench/traces.jsonl}"
HTML_REPORT="${OUTPUT_JSONL%.jsonl}.html"

RUNNER="${NPU_KERNELBENCH_RUNNER:-persistent}"
NPU_IDS="${NPU_KERNELBENCH_NPU_IDS:-0}"
MODE_FLAG=""
if [ -n "${NPU_KERNELBENCH_MODE:-}" ]; then
    MODE_FLAG="--mode ${NPU_KERNELBENCH_MODE}"
fi
LIMIT_FLAG=""
if [ -n "${NPU_KERNELBENCH_LIMIT:-}" ]; then
    LIMIT_FLAG="--limit ${NPU_KERNELBENCH_LIMIT}"
fi
VERBOSE_FLAG=""
if [ -n "${NPU_KERNELBENCH_VERBOSE:-}" ]; then
    VERBOSE_FLAG="--verbose"
fi

cd "${PROJECT_ROOT}"

set -x
uv run npu-kernelbench eval "${PROBLEM_DIR}" \
    --solution "${SOLUTION_JSON}" \
    --runner "${RUNNER}" \
    --npu-ids "${NPU_IDS}" \
    -o "${OUTPUT_JSONL}" \
    --html "${HTML_REPORT}" \
    ${MODE_FLAG} \
    ${LIMIT_FLAG} \
    ${VERBOSE_FLAG}

set +x
echo ""
echo "Traces:  ${OUTPUT_JSONL}"
echo "Report:  ${HTML_REPORT}"
