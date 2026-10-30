#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
echo "current dir: $CURRENT_DIR"

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
fi
echo "ASCEND_INSTALL_PATH: $_ASCEND_INSTALL_PATH"

PROJECT_ROOT=$CURRENT_DIR/../../../
echo "PROJECT_ROOT: $PROJECT_ROOT"

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -rf $PROJECT_ROOT/build
    rm -rf $PROJECT_ROOT/build_out
    rm -rf $CURRENT_DIR/output
    echo "clear success!"

    # 2.编译custom包
    cd $PROJECT_ROOT
    bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_infer_attention_score

    # 3安装custom包
    CUSTOM_PATH=$(realpath "$_ASCEND_INSTALL_PATH/../")
    ./build_out/cann* --install-path=$CUSTOM_PATH

    # 4. source custom包
    source $CUSTOM_PATH/vendors/custom_transformer/bin/set_env.bash

    # 5.运行测试文件（精度和性能）
    cd $CURRENT_DIR
    msprof_path=$(realpath "$_ASCEND_INSTALL_PATH/tools/msopt/bin/msopprof")
    $msprof_path --output=output --aic-metrics=Roofline,Occupancy,Default python3 -m pytest -rA -s ./tests/pytest/test.py -v -m ci
}

main
