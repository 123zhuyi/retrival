#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set -e

SCRIPT_NAME="op_compile_time_grouping"

LOG_FILE=""
CODE_PATH=""

TMP_FILES=(
  result_math.csv
  result_conversion.csv
  result_random.csv
  result0.csv
  result1.csv
)

# =========================================
# Help
# =========================================
show_help() {
cat << EOF
${SCRIPT_NAME}

Description:
  Analyze compile logs, extract each operator's maximum compile time,
  and group operators into 5 groups based on total time proportion.
  Generates result.csv and op_compile_group.yaml.

Usage:
  ${SCRIPT_NAME}.sh -l <log_file> -c <code_path>

Options:
  -l, --log     Compile log file (required)
  -c, --code    Operator source root path (required)
  -h, --help    Show this help message

Example:
  ./${SCRIPT_NAME}.sh -l build.log -c /workspace/op_code
EOF
}

# =========================================
# Argument parsing
# =========================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -c|--code)
                CODE_PATH="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "$LOG_FILE" || -z "$CODE_PATH" ]]; then
        echo "ERROR: --log and --code are required"
        show_help
        exit 1
    fi

    [[ -f "$LOG_FILE" ]] || { echo "ERROR: log file not found: $LOG_FILE"; exit 1; }
    [[ -d "$CODE_PATH" ]] || { echo "ERROR: code path not found: $CODE_PATH"; exit 1; }
}

cleanup() {
    for f in "${TMP_FILES[@]}"; do
        [[ -f "$f" ]] && rm -f "$f"
    done
}
trap cleanup EXIT

generate_yaml() {
    yaml_file="op_compile_group.yaml"
    echo "# Generated from result.csv" > "$yaml_file"

    for g in {1..5}; do
        echo "operator_group_${g}:" >> "$yaml_file"
        awk -F',' -v grp="$g" '$3==grp {print "  - "$1}' result.csv >> "$yaml_file"
    done

    echo "Generate $yaml_file successfully"
}

main() {

math_path="$CODE_PATH/math"
conversion_path="$CODE_PATH/conversion"
random_path="$CODE_PATH/random"

declare -A op_time_max

while read -r op time; do
    prev=${op_time_max[$op]:-0}
    max=$(awk -v a="$prev" -v b="$time" 'BEGIN{print (a>=b)?a:b}')
    op_time_max["$op"]=$max
done < <(
    grep -- '--exe_time=' "$LOG_FILE" | \
    awk '
    {
        op=""
        time=""

        for (i=1; i<=NF; i++) {
            if ($i ~ /--main_func=/) {
                split($i,a,"=")
                op=a[2]
            }
            else if ($i ~ /--exe_time=/) {
                split($i,b,"=")
                time=b[2]
            }
        }

        if (op!="" && time!="")
            print op, time
    }'
)

# safety check
if [[ ${#op_time_max[@]} -eq 0 ]]; then
    echo "ERROR: no operators parsed from log"
    exit 1
fi

total_time=0

for op in "${!op_time_max[@]}"; do
    val="${op_time_max[$op]}"
    base_op="${op%_apt}"

    if [[ -d "$math_path/$base_op" ]]; then
        echo "$base_op,$val" >> result_math.csv
    elif [[ -d "$conversion_path/$base_op" ]]; then
        echo "$base_op,$val" >> result_conversion.csv
    elif [[ -d "$random_path/$base_op" ]]; then
        echo "$base_op,$val" >> result_random.csv
    else
        echo "ERROR: operator not found in any category: $base_op"
        exit 1
    fi

    total_time=$(awk -v a="$total_time" -v b="$val" 'BEGIN{print a+b}')
done

echo "TOTAL_TIME=$total_time"

{
  [[ -f result_math.csv ]] && sort result_math.csv
  [[ -f result_conversion.csv ]] && sort result_conversion.csv
  [[ -f result_random.csv ]] && sort result_random.csv
} > result0.csv

echo "op_name,exe_time_sum" > result1.csv
cat result0.csv >> result1.csv

awk -F',' '
function assign_group(n,    i,sum,threshold,group) {
    threshold = total / 5
    sum = 0
    group = 1
    for (i=2;i<=n;i++) {
        sum += time[i]
        if (sum > group * threshold && group < 5)
            group++
        gid[i]=group
    }
}

NR==1 { header=$0; next }

{
    op[NR]=$1
    time[NR]=$2
    total+=$2
}

END{
    assign_group(NR)
    print header ",group"
    for(i=2;i<=NR;i++)
        print op[i] "," time[i] "," gid[i]
}
' result1.csv > result.csv

echo "Generate result.csv successfully"

generate_yaml
}

parse_args "$@"
main