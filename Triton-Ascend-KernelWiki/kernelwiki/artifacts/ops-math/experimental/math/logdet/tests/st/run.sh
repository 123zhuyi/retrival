#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# logdet 算子 ST 测试执行脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_MOCK=""
BUILD_DIR=""

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock          使用 Mock 模式（CPU golden 验证，无需 NPU）"
    echo "  --real          使用 Real 模式（NPU 执行，默认）"
    echo "  --help          显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # C++ Mock 模式批量测试（无 NPU 环境）"
    echo "  $0 --mock"
    echo ""
    echo "  # C++ Real 模式批量测试（需要 NPU）"
    echo "  $0"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)
            USE_MOCK="-DUSE_MOCK=ON"
            shift
            ;;
        --real)
            USE_MOCK=""
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
fi

echo "========================================"
echo "logdet 算子 ST 测试 (C++)"
echo "========================================"
if [ -n "$USE_MOCK" ]; then
    echo "模式: Mock (CPU golden 验证)"
else
    echo "模式: Real (NPU 执行，默认)"
fi
echo "工作目录: ${SCRIPT_DIR}"
echo "========================================"
echo ""

echo "检查依赖..."

if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi

if [ -z "$USE_MOCK" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
        echo "建议设置: source set_env.sh (in CANN install directory)"
    fi
fi

echo "依赖检查完成"
echo ""

if [ -z "$USE_MOCK" ]; then
    echo "设置环境变量..."

    if [ -n "$ASCEND_HOME_PATH" ]; then
        source "${ASCEND_HOME_PATH}/set_env.sh"
        echo "已加载 CANN 环境: ${ASCEND_HOME_PATH}"
    else
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
    fi

    echo "环境变量设置完成"
    echo ""
fi

echo "创建构建目录..."
mkdir -p "${BUILD_DIR}"

echo ""
echo "CMake 配置..."
cd "${BUILD_DIR}"

if [ -n "$USE_MOCK" ]; then
    cmake .. -DUSE_MOCK=ON
else
    cmake ..
fi

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    exit 1
fi

echo ""
echo "编译测试程序..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

echo "编译成功"
echo ""

echo "========================================"
echo "执行测试"
echo "========================================"

./test_aclnn_logdet

TEST_RESULT=$?

echo ""
echo "========================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "测试结果: PASS ✓"
else
    echo "测试结果: FAIL ✗"
fi
echo "========================================"
echo ""

exit $TEST_RESULT
