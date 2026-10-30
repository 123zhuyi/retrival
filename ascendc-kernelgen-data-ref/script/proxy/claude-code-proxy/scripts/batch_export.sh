#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# batch_export.sh — 批量按算子导出轨迹
#
# 遍历 cc_traces/*.db（跳过默认 trace.db），对每个算子 db 调 export_trajectories.py
# 和/或 export_training.py，实现一键导出所有算子的轨迹/训练数据。
#
# 用法：
#   bash scripts/batch_export.sh [trace_dir] [out_dir] [trajectories|training|both]
#
# 参数：
#   trace_dir  存放 *.db 的目录，默认 脚本同目录/../cc_traces
#   out_dir    输出根目录，默认 脚本同目录/../export_out
#   mode       trajectories（仅轨迹）/ training（仅训练数据）/ both（两者，默认）
#
# 示例：
#   bash scripts/batch_export.sh
#   bash scripts/batch_export.sh /path/to/cc_traces /path/to/out training
# ----------------------------------------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACE_DIR="${1:-${SCRIPT_DIR}/../cc_traces}"
OUT_DIR="${2:-${SCRIPT_DIR}/../export_out}"
MODE="${3:-both}"

if [[ ! -d "$TRACE_DIR" ]]; then
    echo "错误：trace 目录不存在：$TRACE_DIR" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

db_count=0
ok_traj=0
ok_train=0
fail_traj=0
fail_train=0

for db in "$TRACE_DIR"/*.db; do
    [[ -f "$db" ]] || continue
    name=$(basename "$db" .db)
    # 跳过默认 trace.db（fallback 数据，非算子）
    [[ "$name" == "trace" ]] && continue
    db_count=$((db_count + 1))
    echo ">>> [$db_count] $name"

    if [[ "$MODE" == "trajectories" || "$MODE" == "both" ]]; then
        if python3 "${SCRIPT_DIR}/export_trajectories.py" \
            --db "$db" --out "${OUT_DIR}/trajectories" >/dev/null 2>&1; then
            echo "    trajectories OK"
            ok_traj=$((ok_traj + 1))
        else
            echo "    trajectories 失败"
            fail_traj=$((fail_traj + 1))
        fi
    fi
    if [[ "$MODE" == "training" || "$MODE" == "both" ]]; then
        if python3 "${SCRIPT_DIR}/export_training.py" \
            --db "$db" --out "${OUT_DIR}/training" >/dev/null 2>&1; then
            echo "    training OK"
            ok_train=$((ok_train + 1))
        else
            echo "    training 失败"
            fail_train=$((fail_train + 1))
        fi
    fi
done

echo ""
echo "================================================================"
echo "批量导出完成"
echo "  处理算子 db 数：$db_count"
echo "  trajectories：成功 $ok_traj，失败 $fail_traj"
echo "  training：     成功 $ok_train，失败 $fail_train"
echo "  输出目录：     $OUT_DIR"
echo "================================================================"