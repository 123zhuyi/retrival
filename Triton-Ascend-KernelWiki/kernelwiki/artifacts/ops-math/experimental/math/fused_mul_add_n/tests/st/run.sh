#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# FusedMulAddN 算子 C++ ST 测试执行脚本
#
# 公式: y_i = x1_i * x3[0] + x2_i
# 接口形态: 本 C++ ST 以 Mock 模式（CPU golden 自测）验证测试框架与公式实现。
#           设备侧上板精度验收由独立 PyTorch（torch_npu）任务承担。
# 用例覆盖: L0 + L1 多 shape，以及边界不变量(x3=0/1/负)、
#           极端输入(NaN/Inf/全零/fp16上界)、整数回绕、广播等价、确定性。
#
# 用法:
#   bash run.sh          # 默认 = Mock 模式（CPU golden，无需 NPU）
#   bash run.sh --mock   # 显式 Mock 模式
#   bash run.sh --real   # Real 编译（仍执行 CPU golden 自测；无 aclnn 链接）
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 默认 Mock（Mock 为已验证的 CPU golden 路径）
USE_MOCK="-DUSE_MOCK=ON"
BUILD_DIR=""

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock          使用 Mock 模式（CPU golden 验证，无需 NPU，默认）"
    echo "  --real          使用 Real 编译（本算子无 aclnn，仍执行 CPU golden 自测）"
    echo "  --help          显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0            # 默认 Mock 模式（CPU golden 自测）"
    echo "  $0 --mock     # 显式 Mock 模式"
    echo "  $0 --real     # Real 编译"
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

# 根据模式设置独立构建目录，避免 mock/real 缓存冲突
if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
fi

echo "========================================"
echo "FusedMulAddN 算子 ST 测试 (C++)"
echo "========================================"
if [ -n "$USE_MOCK" ]; then
    echo "模式: Mock (CPU golden 验证)"
else
    echo "模式: Real (本算子无 aclnn，执行 CPU golden 自测)"
fi
echo "工作目录: ${SCRIPT_DIR}"
echo "========================================"
echo ""

# ============================================================================
# 检查依赖
# ============================================================================
echo "检查依赖..."
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi
echo "依赖检查完成"
echo ""

# ============================================================================
# 创建构建目录 + CMake 配置
# ============================================================================
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

# ============================================================================
# 编译
# ============================================================================
echo ""
echo "编译测试程序..."
make -j"$(nproc)"
echo "编译成功"
echo ""

# ============================================================================
# 执行测试
# ============================================================================
echo "========================================"
echo "执行测试"
echo "========================================"
./test_aclnn_fused_mul_add_n
TEST_RESULT=$?

echo ""
echo "========================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "测试结果: PASS"
else
    echo "测试结果: FAIL"
fi
echo "========================================"
echo ""

exit $TEST_RESULT
