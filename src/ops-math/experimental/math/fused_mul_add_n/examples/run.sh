#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
#
# FusedMulAddN examples 一键编译 + 运行脚本（真实 NPU / ascend910b）
#
# 覆盖两类调用入口：
#   - aclnn  : test_aclnn_fused_mul_add_n.cpp  （aclnn 两段式，链接自定义包 libcust_opapi.so）
#   - geir   : test_geir_fused_mul_add_n.cpp   （图模式 GE IR 构图，编译验证）
#
# 用法:
#   bash run.sh              # 默认编译并运行 aclnn 示例 + 编译 geir 示例
#   bash run.sh aclnn        # 仅 aclnn 示例（编译 + 运行）
#   bash run.sh geir         # 仅 geir 示例（编译；如有 NPU 可运行）
#   bash run.sh --noexec     # 仅编译，不运行
#
# 前置：
#   1) source <CANN>/set_env.sh （设置 ASCEND_HOME_PATH / ASCEND_OPP_PATH）
#   2) aclnn 示例需自定义算子包 custom_math（含 aclnnFusedMulAddN + ascend910b kernel）。
#      自定义包查找优先级：
#        a) 环境变量 CUSTOM_OP_VENDOR_DIR 显式指定（指向 .../vendors/custom_math）
#        b) 仓库内 build_out/installed_custom_math/packages/vendors/custom_math
#        c) $ASCEND_HOME_PATH/opp/vendors/custom_math 或 $HOME/Ascend/opp/vendors/custom_math
#      若未生成，请先构建并解包：
#        bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_mul_add_n
#        bash build_out/cann-ops-math-custom_linux-*.run --noexec --quiet \
#             --extract=$(pwd)/build_out/installed_custom_math
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 解析参数 ---
TARGET="all"
NOEXEC="false"
for arg in "$@"; do
    case "$arg" in
        aclnn|geir|all) TARGET="$arg" ;;
        --noexec)       NOEXEC="true" ;;
        *) echo "[WARN] 未知参数: $arg（可用: aclnn | geir | all | --noexec）" ;;
    esac
done

# --- 校验 CANN 环境 ---
if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置或不存在，请先 source <CANN>/set_env.sh。"
    exit 1
fi
ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}"

# CANN 路径（与仓库 build.sh examples 编译保持一致）
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
GRAPH_INCLUDE_PATH="${INCLUDE_PATH}/graph"
GE_INCLUDE_PATH="${INCLUDE_PATH}/ge"
GE_EXTERNAL_INCLUDE_PATH="${INCLUDE_PATH}/external"
INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

# 定位仓库根（worktree 根）：examples -> 上溯到含 build.sh 的目录
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/build.sh" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done

resolve_vendor() {
    local candidates=(
        "${CUSTOM_OP_VENDOR_DIR:-}"
        "$REPO_ROOT/build_out/installed_custom_math/packages/vendors/custom_math"
        "${ASCEND_HOME_PATH}/opp/vendors/custom_math"
        "$HOME/Ascend/opp/vendors/custom_math"
    )
    for c in "${candidates[@]}"; do
        if [[ -n "$c" && -f "$c/op_api/include/aclnn_fused_mul_add_n.h" \
              && -f "$c/op_api/lib/libcust_opapi.so" ]]; then
            echo "$c"; return 0
        fi
    done
    return 1
}

build_run_aclnn() {
    echo "=========================================================="
    echo "[aclnn] 编译 test_aclnn_fused_mul_add_n.cpp（自定义包两段式）"
    local vendor
    vendor="$(resolve_vendor || true)"
    if [[ -z "$vendor" ]]; then
        echo "[ERROR] 未找到 custom_math 自定义算子包（含 aclnn_fused_mul_add_n.h + libcust_opapi.so）。"
        echo "        请先构建并解包，或设置 CUSTOM_OP_VENDOR_DIR。详见脚本头部注释。"
        return 2
    fi
    echo "[aclnn] custom_math vendor dir: $vendor"

    g++ test_aclnn_fused_mul_add_n.cpp \
        -std=c++17 \
        -I "${vendor}/op_api/include" \
        -I "${INCLUDE_PATH}" \
        -I "${INCLUDE_PATH}/aclnnop" \
        -I "${ACLNN_INCLUDE_PATH}" \
        -L "${vendor}/op_api/lib" \
        -L "${EAGER_LIBRARY_PATH}" \
        -lcust_opapi -lascendcl -lnnopbase \
        -o test_aclnn_fused_mul_add_n \
        -Wl,-rpath="${vendor}/op_api/lib:${EAGER_LIBRARY_PATH}"
    echo "[aclnn] 编译成功 -> ./test_aclnn_fused_mul_add_n"

    if [[ "$NOEXEC" == "true" ]]; then
        echo "[aclnn] --noexec 指定，跳过运行。"
        return 0
    fi
    # 运行时让 aclnn 找到自定义 kernel
    export ASCEND_CUSTOM_OPP_PATH="${vendor}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="${vendor}/op_api/lib:${EAGER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[aclnn] ASCEND_CUSTOM_OPP_PATH=$ASCEND_CUSTOM_OPP_PATH"
    echo "[aclnn] 运行（期望 result = {3,5,7,9,11,13,15,17}）:"
    ./test_aclnn_fused_mul_add_n
}

build_run_geir() {
    echo "=========================================================="
    echo "[geir] 编译 test_geir_fused_mul_add_n.cpp（图模式 GE IR）"
    g++ test_geir_fused_mul_add_n.cpp \
        -std=c++17 \
        -I "${GE_EXTERNAL_INCLUDE_PATH}" \
        -I "${GRAPH_INCLUDE_PATH}" \
        -I "${GE_INCLUDE_PATH}" \
        -I "${INCLUDE_PATH}" \
        -I "${INC_INCLUDE_PATH}" \
        -L "${GRAPH_LIBRARY_PATH}" \
        -lgraph -lge_runner -lgraph_base -lge_compiler \
        -o test_geir_fused_mul_add_n \
        -Wl,-rpath="${GRAPH_LIBRARY_PATH}"
    echo "[geir] 编译成功 -> ./test_geir_fused_mul_add_n"

    if [[ "$NOEXEC" == "true" ]]; then
        echo "[geir] --noexec 指定，跳过运行。"
        return 0
    fi
    # 图模式运行需自定义算子包提供 op_proto / kernel。若无则仅完成编译验证。
    local vendor
    vendor="$(resolve_vendor || true)"
    if [[ -n "$vendor" ]]; then
        export ASCEND_CUSTOM_OPP_PATH="${vendor}${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
        export LD_LIBRARY_PATH="${GRAPH_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[geir] 运行（期望 result = 6 = 2*2+2）:"
        ./test_geir_fused_mul_add_n || echo "[geir] 运行失败（图模式上板依赖自定义包注册，编译已通过）。"
    else
        echo "[geir] 未找到自定义包，跳过运行（编译已通过）。"
    fi
}

case "$TARGET" in
    aclnn) build_run_aclnn ;;
    geir)  build_run_geir ;;
    all)   build_run_aclnn; build_run_geir ;;
esac
echo "=========================================================="
echo "[DONE] examples 处理完成。"
