#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
# 输入选项默认初值
BS=64
HIDDEN_SIZE=8192
RUN_TYPE=0
SCALES_TYPE="float32_t"
OUTPUT_TYPE="float16_t"
MXFP=0
CASE_NAME="int8_t-float32_t-float16_t-ID001"
RANK_SIZE=2

SHORT=b:,h:,t:,s:,o:,m:,n:,d:,
LONG=bs:,hidden-size:,run-type:,scales-type:,output_type:,mxfp:,case-name:,rank-size:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -b | --bs)
        BS="$2"
        shift 2
        ;;
    -h | --hidden-size)
        HIDDEN_SIZE="$2"
        shift 2
        ;;
    -t | --run-type)
        RUN_TYPE="$2"
        shift 2
        ;;
    -s | --scales-type)
        SCALES_TYPE="$2"
        shift 2
        ;;
    -o | --output_type):
        OUTPUT_TYPE="$2"
        shift 2
        ;;
    -m | --mxfp):
        MXFP="$2"
        shift 2
        ;;
    -n | --case-name):
        CASE_NAME="$2"
        shift 2
        ;;   
    -d | --rank-size):
        RANK_SIZE="$2"
        shift 2
        ;;     
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done
# 导入环境变量
export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
echo "========== 开始执行程序 =========="
echo "RANKSIZE 大小 "$RANK_SIZE""
RANKSIZE=$((RANK_SIZE))
for ((rank_id=0; rank_id<RANKSIZE; rank_id++))
do
    (
        nohup ./out/bin/test_quant_all_reduce \
            --rank_id ${rank_id} \
            --bs ${BS} \
            --hidden_size ${HIDDEN_SIZE} \
            --run_type ${RUN_TYPE} \
            --scales_type ${SCALES_TYPE} \
            --output_type ${OUTPUT_TYPE} \
            --mxfp ${MXFP} \
            --case_name ${CASE_NAME} \
    ) &
done

wait
echo "========== 程序执行完成 =========="