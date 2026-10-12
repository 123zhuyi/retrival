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
BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"
TARGET_LINE=2
SOC_VERSION="Ascend310P3"
PROFING_TYPE="OFF"

SHORT=c:,v:,i:,b:,p:,r:,n:,f:,
LONG=cmake-rebuild:,soc-version:,install-path:,build-type:,install-prefix:,run-test:,target-line:,profing-type:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -c | --cmake-rebuild):
        CMAKE_REBUILD="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    -r | --run-test):
        RUN_TEST="$2"
        shift 2
        ;;
    -n | --target-line):
        TARGET_LINE="$2"
        shift 2
        ;;
    -f | --profing-type):
        PROFING_TYPE="$2"
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
# 编译芯片型号选择
VERSION_LIST="ascend950pr_9599"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi
# 获取环境变量ASCEND_HOME_PATH
if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
# 设置环境变量
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

set -e
# 编译开关
if [[ "$CMAKE_REBUILD" == "ON" ]]; then
    rm -rf build
    mkdir -p build
    cmake -B build \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
    cmake --build build -j
    cmake --install build
fi
# 运行用例开关
if [[ "$RUN_TEST" == "ON" ]]; then    
    # 读取目录下QuantAllReduce配置文件
    CSV_FILE="${CURRENT_DIR}/scripts/QuantAllReduce.csv"
    target_content=$(sed -n "${TARGET_LINE}p" "$CSV_FILE")
    IFS=',' read -r \
        case_name \
        case_type \
        bs \
        hidden_size \
        input_tensor_range \
        input_tensor_type \
        scales_type \
        scales_range \
        output_type \
        ranksize \
        reduce_op \
        mxfp \
        seed \
    <<< "$target_content"    
    echo "===== 执行第${TARGET_LINE}条用例 ====="
    echo "case_name : $case_name"
    echo "case_type : $case_type"
    echo "bs : $bs"
    echo "hidden_size : $hidden_size"
    echo "input_tensor_range : $input_tensor_range"
    echo "input_tensor_type : $input_tensor_type"
    echo "scales_range : $scales_range"
    echo "scales_type : $scales_type"
    echo "output_type : $output_type"
    echo "ranksize : $ranksize"
    echo "reduce_op : $reduce_op"
    echo "seed : $seed"
    echo "mxfp : $mxfp"
    echo "======================================"
    # 运行数据生成脚本
    python3 ${CURRENT_DIR}/scripts/data_generate.py \
            "$case_name" \
            "$bs" \
            "$hidden_size" \
            "$input_tensor_range" \
            "$input_tensor_type" \
            "$scales_range" \
            "$scales_type" \
            "$output_type" \
            "$ranksize" \
            "$reduce_op" \
            "$mxfp" \
            "$seed"
    chmod +x ${CURRENT_DIR}/start.sh
    # profing开关
    if [[ "$PROFING_TYPE" == "ON" ]]; then
        # 采集性能数据
        msprof \
            --application="./start.sh -b $bs -h $hidden_size -t $case_type -s $scales_type -o $output_type -m $mxfp -n $case_name -d $ranksize" \
            --output=./prof
    else
        # 直接运行不采集性能数据
        ./start.sh -b $bs -h $hidden_size -t $case_type -s $scales_type -o $output_type -m $mxfp -n $case_name -d $ranksize
    fi
    # 精度对比
    python3 ${CURRENT_DIR}/scripts/data_compare.py \
            "$case_name" \
            "$bs" \
            "$hidden_size" \
            "$output_type" \
            "$ranksize" 
fi