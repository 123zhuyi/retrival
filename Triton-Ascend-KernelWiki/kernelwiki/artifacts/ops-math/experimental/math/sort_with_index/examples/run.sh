#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# SortWithIndex (experimental, ascend910b) examples 一键编译运行脚本。
#
# 默认编译 aclnn + geir 两个示例，并在真实 NPU 上运行 aclnn 升序基础用例。
#
# 前置：已安装本算子的独立 vendor 自定义算子包（规避 CANN 自带系统 built-in SortWithIndex）：
#   bash build.sh --pkg --experimental --soc=ascend910b --ops=sort_with_index \
#        --vendor_name=sort_with_index_custom -j16
#   ./build_out/cann-ops-math-sort_with_index_custom_linux-*.run --install-path="${HOME}/sort_with_index_opp" --quiet
# 若已 export ASCEND_CUSTOM_OPP_PATH 指向自定义算子包 vendor 根，本脚本直接使用；否则按默认安装路径探测，
# 并设置 ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH。
#
# 用法:
#   bash run.sh              # 编译 + 运行 aclnn 示例（默认，真实 NPU）
#   bash run.sh --geir       # 额外运行 geir 示例（需 GE 图执行环境，构图必通过）
#   bash run.sh --no-run     # 仅编译（不运行）
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RUN_ACLNN=1
RUN_GEIR=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --geir)   RUN_GEIR=1; shift ;;
        --no-run) RUN_ACLNN=0; RUN_GEIR=0; shift ;;
        --help|-h)
            sed -n '12,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ============================================================================
# 1. CANN 环境
# ============================================================================
if [ -z "${ASCEND_HOME_PATH}" ]; then
    for set_env in \
        /usr/local/Ascend/cann-9.0.0/set_env.sh \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        "${HOME}/Ascend/ascend-toolkit/set_env.sh"; do
        if [ -f "${set_env}" ]; then
            source "${set_env}"
            break
        fi
    done
fi
if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "错误: ASCEND_HOME_PATH 未设置，请先 source CANN 的 set_env.sh"
    exit 1
fi
echo "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"

# ============================================================================
# 2. 定位自定义算子包 vendor（规避系统 built-in SortWithIndex）
#    vendor 名 = sort_with_index_custom_math（--vendor_name=sort_with_index_custom 安装后的 vendor 目录）。
#    若已 export ASCEND_CUSTOM_OPP_PATH 则直接使用；否则按默认安装路径探测。
#    SWI_OPP_INSTALL_PATH 可覆盖默认的算子包安装根（与 --install-path 一致）。
# ============================================================================
SWI_OPP_INSTALL_PATH="${SWI_OPP_INSTALL_PATH:-${HOME}/sort_with_index_opp}"
if [ -z "${ASCEND_CUSTOM_OPP_PATH}" ]; then
    for vroot in \
        "${SWI_OPP_INSTALL_PATH}/vendors/sort_with_index_custom_math" \
        "${ASCEND_HOME_PATH}/opp/vendors/sort_with_index_custom_math" \
        "${ASCEND_HOME_PATH}/opp/vendors/custom_math"; do
        if [ -d "${vroot}" ]; then
            export ASCEND_CUSTOM_OPP_PATH="${vroot}"
            break
        fi
    done
fi
if [ -n "${ASCEND_CUSTOM_OPP_PATH}" ]; then
    echo "ASCEND_CUSTOM_OPP_PATH = ${ASCEND_CUSTOM_OPP_PATH}"
    # 优先加载 vendor 的 set_env.bash（若存在），并把 op_api/lib 加入 LD_LIBRARY_PATH。
    VENDOR_ROOT="${ASCEND_CUSTOM_OPP_PATH%%:*}"
    if [ -f "${VENDOR_ROOT}/bin/set_env.bash" ]; then
        source "${VENDOR_ROOT}/bin/set_env.bash" || true
    fi
    export LD_LIBRARY_PATH="${VENDOR_ROOT}/op_api/lib:${LD_LIBRARY_PATH}"
else
    echo "警告: 未探测到自定义算子包 vendor（sort_with_index_custom_math）。"
    echo "      请先构建并安装算子包（见脚本头注释），否则 aclnn 示例链接/运行会失败。"
fi
echo ""

# ============================================================================
# 3. 编译
# ============================================================================
echo "=========================================="
echo "编译 SortWithIndex examples"
echo "=========================================="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. >/dev/null
make -j"$(nproc)"
echo "编译完成"
echo ""

# ============================================================================
# 4. 运行 aclnn 示例（真实 NPU）
# ============================================================================
RESULT=0
if [ "${RUN_ACLNN}" -eq 1 ]; then
    if [ -x "${BUILD_DIR}/test_aclnn_sort_with_index" ]; then
        echo "=========================================="
        echo "运行 aclnn 示例（真实 NPU，升序基础用例）"
        echo "=========================================="
        "${BUILD_DIR}/test_aclnn_sort_with_index"
        RESULT=$?
        echo ""
    else
        echo "错误: 未生成 test_aclnn_sort_with_index（检查自定义算子包安装）"
        RESULT=1
    fi
fi

# ============================================================================
# 5. （可选）运行 geir 示例
# ============================================================================
if [ "${RUN_GEIR}" -eq 1 ] && [ -x "${BUILD_DIR}/test_geir_sort_with_index" ]; then
    echo "=========================================="
    echo "运行 geir 示例（图模式构图；RunGraph 需图执行环境）"
    echo "=========================================="
    "${BUILD_DIR}/test_geir_sort_with_index" || true
    echo ""
fi

echo "=========================================="
if [ "${RESULT}" -eq 0 ]; then
    echo "示例运行结果: PASS"
else
    echo "示例运行结果: FAIL"
fi
echo "=========================================="
exit ${RESULT}
