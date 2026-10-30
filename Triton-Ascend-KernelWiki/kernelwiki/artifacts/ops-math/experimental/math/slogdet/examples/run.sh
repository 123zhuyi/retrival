#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# slogdet 算子调用示例执行脚本（experimental/math/slogdet）。
#   --eager  : aclnn 两段式调用示例（默认，真机 NPU 运行 + CPU golden 比对）
#   --graph  : 图模式 (GE IR) 构图示例（编译 + 尝试 RunGraph）
#
# 前置条件：
#   - 已 source CANN set_env.sh（ASCEND_HOME_PATH 生效）。
#   - slogdet 自定义算子包已安装，SLOGDET_CUSTOM_OPP 指向其 vendor 目录，例如：
#       export SLOGDET_CUSTOM_OPP=/path/to/vendors/custom_math
#     （若未显式设置，脚本会尝试 ASCEND_HOME_PATH 下的常见 OPP vendor 路径回退查找。）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_MODE="eager"
CLEAN_BUILD=false

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "slogdet 算子调用示例：演示 aclnn / 图模式 (GE IR) 调用方式。"
    echo ""
    echo "Options:"
    echo "  --eager          运行 aclnn 调用示例（默认；真机 NPU + CPU golden 比对）"
    echo "  --graph          运行图模式 (GE IR) 调用示例（编译 + RunGraph）"
    echo "  --clean          清理构建目录后退出"
    echo "  -h, --help       显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0               # aclnn 调用示例（默认）"
    echo "  $0 --graph       # 图模式调用示例"
    echo "  $0 --clean       # 清理构建目录"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --eager) EXAMPLE_MODE="eager"; shift ;;
        --graph) EXAMPLE_MODE="graph"; shift ;;
        --clean) CLEAN_BUILD=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "未知参数: $1"; show_help; exit 1 ;;
    esac
done

if [ "$EXAMPLE_MODE" = "graph" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-geir"
else
    BUILD_DIR="${SCRIPT_DIR}/build-eager"
fi

if [ "$CLEAN_BUILD" = true ]; then
    echo "清理构建目录..."
    rm -rf "${SCRIPT_DIR}/build-eager" "${SCRIPT_DIR}/build-geir"
    echo "清理完成"
    exit 0
fi

echo "========================================"
echo "slogdet 算子调用示例"
echo "========================================"
if [ "$EXAMPLE_MODE" = "graph" ]; then
    echo "模式: 图模式 (GE IR)"
else
    echo "模式: aclnn (eager)"
fi
echo "工作目录: ${SCRIPT_DIR}"
echo "========================================"
echo ""

# ---- 检查依赖 ----
echo "检查依赖..."
command -v cmake >/dev/null 2>&1 || { echo "错误: 未找到 cmake"; exit 1; }
command -v g++ >/dev/null 2>&1 || { echo "错误: 未找到 g++"; exit 1; }
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "警告: 未设置 ASCEND_HOME_PATH，建议先 source CANN set_env.sh"
fi
echo "依赖检查完成"
echo ""

# ---- 定位 slogdet 自定义算子 vendor 目录并设置环境变量 ----
echo "设置环境变量..."
CUSTOM_OP_VENDOR_DIR=""
for candidate in \
    "${SLOGDET_CUSTOM_OPP}" \
    "${ASCEND_HOME_PATH}/opp/vendors/custom_math" \
    "${ASCEND_HOME_PATH}/opp/vendors/slogdet_custom" \
    "${ASCEND_HOME_PATH}/opp/vendors/customize"; do
    if [ -n "$candidate" ] && [ -f "${candidate}/op_api/include/aclnn_slogdet_native.h" ]; then
        CUSTOM_OP_VENDOR_DIR="$candidate"
        break
    fi
done
if [ -n "$CUSTOM_OP_VENDOR_DIR" ]; then
    export SLOGDET_CUSTOM_OPP="${CUSTOM_OP_VENDOR_DIR}"
    export LD_LIBRARY_PATH="${CUSTOM_OP_VENDOR_DIR}/op_api/lib:${LD_LIBRARY_PATH}"
    # ASCEND_CUSTOM_OPP_PATH 让 runtime 定位自定义 kernel 二进制 (.o/.json)
    export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_OP_VENDOR_DIR}:${ASCEND_CUSTOM_OPP_PATH}"
    echo "自定义算子 vendor 目录: ${CUSTOM_OP_VENDOR_DIR}"
    echo "ASCEND_CUSTOM_OPP_PATH: ${CUSTOM_OP_VENDOR_DIR}"
else
    echo "警告: 未找到 slogdet 自定义算子 vendor 目录（含 op_api/include/aclnn_slogdet_native.h）"
    echo "      请设置 SLOGDET_CUSTOM_OPP 指向已安装的 vendor 目录。"
fi
echo "环境变量设置完成"
echo ""

echo "创建构建目录: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

echo ""
echo "CMake 配置..."
cd "${BUILD_DIR}"

if [ "$EXAMPLE_MODE" = "graph" ]; then
    # cmake 不支持 -f 指定 CMakeLists 文件名，临时替换为 geir 版本后恢复
    _ORIG_CMAKE="${SCRIPT_DIR}/CMakeLists.txt"
    _BACKUP="${SCRIPT_DIR}/.CMakeLists_aclnn.bak"
    mv "${_ORIG_CMAKE}" "${_BACKUP}"
    cp "${SCRIPT_DIR}/CMakeLists_geir.txt" "${_ORIG_CMAKE}"
    trap 'mv "${_BACKUP}" "${_ORIG_CMAKE}" 2>/dev/null' EXIT
    cmake -DCMAKE_CXX_COMPILER=g++ "${SCRIPT_DIR}"
    mv "${_BACKUP}" "${_ORIG_CMAKE}"
    trap - EXIT
else
    cmake -DCMAKE_CXX_COMPILER=g++ "${SCRIPT_DIR}"
fi

echo ""
echo "编译调用示例..."
make -j"$(nproc)"
echo "编译成功"
echo ""

echo "========================================"
echo "执行调用示例"
echo "========================================"

if [ "$EXAMPLE_MODE" = "graph" ]; then
    echo ""
    echo ">>> 运行图模式 (GE IR) 调用示例 <<<"
    echo ""
    ./test_geir_slogdet 0
else
    echo ""
    echo ">>> 运行 aclnn 调用示例 <<<"
    echo ""
    ./bin/test_aclnn_slogdet
fi
EXAMPLE_RESULT=$?

echo ""
echo "========================================"
if [ $EXAMPLE_RESULT -eq 0 ]; then
    echo "执行结果: PASS"
else
    echo "执行结果: FAIL"
fi
echo "========================================"
echo ""

exit $EXAMPLE_RESULT
