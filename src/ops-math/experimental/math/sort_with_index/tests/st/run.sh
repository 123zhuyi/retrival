#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# sort_with_index 算子 C++ ST 测试执行脚本（ascend910b）
#
# 用法:
#   bash run.sh           # Real 模式（NPU 执行 + bitwise/精确比对，需算子 op_api 已安装）
#   bash run.sh --mock    # Mock 模式（CPU golden 自验证，无需 NPU）
#
# 说明: 本工程为 C++ ST（全量用例），范围 = 4 组 dtype（value{fp16,fp32,bf16,int32}
#       × index{int32}；不含 int64-index）+ 多 rank/多轴长/多行分核 shape + 属性（desc/stable/axis）
#       + 边界（rank0/轴长1/空 tensor/shape 不一致/axis 非最后一维）+ extreme（NaN 升序落开头、
#       ±Inf、全零、全相等、±0；NaN 按 isnan 比较）。golden 按「910B NaN 开头」生成。
#       PyTorch 接入测试见 torch/ 子目录（本脚本不含 --torch 分支）。

set -e

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_MOCK=""  # 默认 Real 模式（NPU）

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock          使用 Mock 模式（CPU golden 自验证，无需 NPU）"
    echo "  --real          使用 Real 模式（NPU 执行，默认）"
    echo "  --help          显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0              # C++ Real 模式（默认，需要 NPU + 已安装 op_api）"
    echo "  $0 --mock       # C++ Mock 模式（无 NPU 环境，验证 golden + 比对闭环）"
}

# ============================================================================
# 解析参数
# ============================================================================
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

# ============================================================================
# 构建目录（按模式隔离，避免 mock/real 缓存冲突）
# ============================================================================
if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
fi

# ============================================================================
# 显示配置信息
# ============================================================================
echo "========================================"
echo "sort_with_index 算子 ST 测试 (C++)"
echo "========================================"
if [ -n "$USE_MOCK" ]; then
    echo "模式: Mock (CPU golden 自验证)"
else
    echo "模式: Real (NPU 执行，默认)"
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
if [ -z "$USE_MOCK" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
        echo "建议设置: source \$ASCEND_HOME_PATH/../set_env.sh (CANN 安装目录下的 set_env.sh)"
    fi
fi
echo "依赖检查完成"
echo ""

# ============================================================================
# 设置环境变量（Real 模式：查找自定义算子 op_api lib）
# ============================================================================
if [ -z "$USE_MOCK" ]; then
    echo "设置环境变量..."
    # The experimental --pkg custom package vendor name is "custom_math". It installs either under
    # $ASCEND_HOME_PATH/opp/vendors (system path) or, when that is not writable, under a user path
    # exported via ASCEND_CUSTOM_OPP_PATH (.../vendors/custom_math). Cover both, plus legacy names.
    CUSTOM_OP_LIB_DIR=""
    for candidate_dir in \
        "${ASCEND_CUSTOM_OPP_PATH%%:*}/op_api/lib" \
        "${ASCEND_HOME_PATH}/opp/vendors/custom_math/op_api/lib" \
        "${ASCEND_HOME_PATH}/opp/vendors/sort_with_index_custom/op_api/lib" \
        "${ASCEND_HOME_PATH}/opp/vendors/customize/op_api/lib" \
        "${HOME}/sort_with_index_custom_opp/vendors/custom_math/op_api/lib" \
        "/usr/local/Ascend/opp/vendors/sort_with_index_custom/op_api/lib" \
        "${HOME}/Ascend/opp/vendors/sort_with_index_custom/op_api/lib"; do
        if [ -d "$candidate_dir" ]; then
            CUSTOM_OP_LIB_DIR="$candidate_dir"
            break
        fi
    done
    if [ -n "$CUSTOM_OP_LIB_DIR" ]; then
        export LD_LIBRARY_PATH=${CUSTOM_OP_LIB_DIR}:${LD_LIBRARY_PATH}
        echo "LD_LIBRARY_PATH: ${CUSTOM_OP_LIB_DIR}"
    else
        echo "警告: 未找到自定义算子包 op_api lib 目录（custom_math/sort_with_index_custom）；请先安装算子包并 source 其 set_env.bash，否则 Real 模式将链接失败"
    fi
    echo "环境变量设置完成"
    echo ""
fi

# ============================================================================
# 创建构建目录
# ============================================================================
echo "创建构建目录..."
mkdir -p "${BUILD_DIR}"

# ============================================================================
# CMake 配置
# ============================================================================
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

# ============================================================================
# 编译
# ============================================================================
echo ""
echo "编译测试程序..."
make -j"$(nproc)"
if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi
echo "编译成功"
echo ""

# ============================================================================
# 执行测试
# ============================================================================
echo "========================================"
echo "执行测试"
echo "========================================"
./test_aclnn_sort_with_index
TEST_RESULT=$?

# ============================================================================
# 输出结果
# ============================================================================
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
