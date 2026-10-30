#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# clean_pipeline.sh — 一键清洗 trace.db → SFT 训练数据
#
# 用法：
#   bash clean_pipeline.sh <trace目录> <算子产出目录> <输出目录>
#
# 环境变量：
#   ASCEND_RT_VISIBLE_DEVICES  NPU 卡号（batch_evaluate 评测用）
#   CLEAN_JOBS                 并发数（默认 4）
# ----------------------------------------------------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLEAN_SCRIPTS="${SCRIPT_DIR}/clean"

TRACE_DIR="${1:?用法: bash clean_pipeline.sh <trace目录> <算子产出目录> <输出目录>}"
OPS_DIR="${2:?缺少算子产出目录}"
OUT_DIR="${3:?缺少输出目录}"

JOBS="${CLEAN_JOBS:-4}"
WORK_DIR=$(mktemp -d
)
mkdir -p "$OUT_DIR/pass" "$OUT_DIR/fail"

# 统计算子数
db_count=$(ls "$TRACE_DIR"/*.db 2>/dev/null | grep -v "trace.db" | wc -l)

echo "[clean] 清洗启动（$db_count 个算子，NPU=${ASCEND_RT_VISIBLE_DEVICES:-未指定}，jobs=$JOBS）"

# ── ① export_training ──
export_ok=0
for db in "$TRACE_DIR"/*.db; do
    [[ -f "$db" ]] || continue
    [[ "$(basename "$db")" == "trace.db" ]] && continue
    op_name="$(basename "$db" .db)"
    python3 "$CLEAN_SCRIPTS/export_training.py" \
        --db "$db" --out "$WORK_DIR/raw/$op_name/" \
        --export-subagents --name-by opsname >> "$OUT_DIR/clean.log" 2>&1 \
        && export_ok=$((export_ok + 1))
done
echo "[clean] ① export_training...        ✅ $export_ok/$db_count"

# ── ② prepare_for_training ──
python3 "$CLEAN_SCRIPTS/prepare_for_training.py" "$WORK_DIR/raw/" >> "$OUT_DIR/clean.log" 2>&1
echo "[clean] ② prepare...                ✅"

# ── ③ batch_evaluate ──
EVAL_NPU_LIST="${EVAL_NPU_LIST:-}" python3 "$CLEAN_SCRIPTS/batch_evaluate.py" \
    "$OPS_DIR" -o "$OUT_DIR/eval_report.json" \
    -j "$JOBS" --timeout 300 >> "$OUT_DIR/clean.log" 2>&1
eval_summary=$(python3 -c "
import json; d=json.load(open('$OUT_DIR/eval_report.json'))
s=d.get('summary',d) if isinstance(d,dict) else {}
print(f\"PASS={s.get('pass',0)} FAIL={s.get('fail',0)} ERROR={s.get('error',0)}\")
" 2>/dev/null || echo "读取失败")
echo "[clean] ③ batch_evaluate...         ✅ $eval_summary"

# ── ④ check_dataset ──
python3 "$CLEAN_SCRIPTS/check_dataset.py" \
    "$WORK_DIR/raw/" --recursive \
    --split "$OUT_DIR/" \
    --eval-report "$OUT_DIR/eval_report.json" \
    -o "$OUT_DIR/check_report.json" >> "$OUT_DIR/clean.log" 2>&1
pass_count=$(ls "$OUT_DIR/pass/"*.json 2>/dev/null | wc -l)
fail_count=$(ls "$OUT_DIR/fail/"*.json 2>/dev/null | wc -l)
echo "[clean] ④ check_dataset...          ✅ pass=$pass_count, fail=$fail_count"

# ── 汇总 ──
echo "[clean] 清洗完成：PASS=$pass_count, FAIL=$fail_count（合格数据在 $OUT_DIR/pass/）"

rm -rf "$WORK_DIR"
