#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
# BitwiseNot (experimental/math/bitwise_not) 调用示例一键编译运行脚本。
#
#   bash run.sh                 # 编译并运行 eager 示例（默认，真实 NPU，两段式 aclnnBitwiseNot）；编译 geir 示例
#   bash run.sh --eager-only    # 仅 eager 示例（编译 + 运行）
#   bash run.sh --geir-build-only  # 仅编译 geir 示例（不实跑图，用于纯编译验收）
#   bash run.sh --run-geir      # 额外实跑 geir 示例（需完整 GE 图执行环境）
#
# 本脚本封装编译运行流程:
#   1) 把自定义算子包安装到 CUSTOM_OPP_DIR（默认 /tmp/bnot_custom_opp，可配置）；
#   2) 创建并 export ASCEND_CACHE_PATH 为算子编译缓存目录（默认 /tmp/bnot_acl_cache，可配置）；
#   3) 从运行目录运行可执行（默认 /tmp/bnot_example_run，可配置）；
#      （示例本身在 Init/load 时 aclSetCompileopt(ACL_OP_JIT_COMPILE,"disable") 使用预编译 binary kernel。）
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根（experimental/math/bitwise_not/examples -> 上溯 4 级）。
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

EAGER_ONLY="OFF"
GEIR_BUILD_ONLY="OFF"
RUN_GEIR="OFF"
for arg in "$@"; do
    case "${arg}" in
        --eager-only)     EAGER_ONLY="ON" ;;
        --geir-build-only) GEIR_BUILD_ONLY="ON" ;;
        --run-geir)       RUN_GEIR="ON" ;;
        *) echo "unknown arg: ${arg}" ;;
    esac
done

# ---- CANN 环境 ----
if [ -z "${ASCEND_HOME_PATH}" ]; then
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
    fi
fi
if [ -d /usr/local/Ascend/cann-9.0.0/bin ]; then
    export PATH=/usr/local/Ascend/cann-9.0.0/bin:${PATH}
fi

# ---- 算子包安装路径 / acl 缓存 / 运行目录（均可通过环境变量配置）----
CUSTOM_OPP_DIR="${BNOT_CUSTOM_OPP_DIR:-/tmp/bnot_custom_opp}"
ACL_CACHE_DIR="${BNOT_ACL_CACHE_DIR:-/tmp/bnot_acl_cache}"
RUN_DIR="${BNOT_EXAMPLE_RUN_DIR:-/tmp/bnot_example_run}"
mkdir -p "${ACL_CACHE_DIR}" "${RUN_DIR}"
export ASCEND_CACHE_PATH="${ACL_CACHE_DIR}"   # 算子编译缓存目录

# ---- 安装自定义算子包（若已装则跳过）----
install_custom_pkg() {
    if [ -f "${CUSTOM_OPP_DIR}/vendors/custom_math/bin/set_env.bash" ]; then
        echo "[run.sh] custom opp already installed at ${CUSTOM_OPP_DIR}"
    else
        local run_pkg
        run_pkg="$(ls "${REPO_ROOT}"/build_out/cann-ops-math-custom_linux-*.run 2>/dev/null | head -n1 || true)"
        if [ -z "${run_pkg}" ]; then
            echo "[run.sh] ERROR: 未找到自定义算子包 build_out/cann-ops-math-custom_linux-*.run。"
            echo "[run.sh]        请先编译: bash build.sh --pkg --experimental --soc=ascend910b --ops=bitwise_not"
            exit 1
        fi
        echo "[run.sh] installing ${run_pkg} -> ${CUSTOM_OPP_DIR}"
        bash "${run_pkg}" --quiet --install-path="${CUSTOM_OPP_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${CUSTOM_OPP_DIR}/vendors/custom_math/bin/set_env.bash"
}

# ---- 若要编译 eager 示例（两段式 aclnnBitwiseNot），须先安装自定义包以提供 libcust_opapi + 头 ----
#      （CMake 据 ASCEND_CUSTOM_OPP_PATH 定位 op_api/include 与 libcust_opapi.so；故在 cmake 前安装）。
if [ "${GEIR_BUILD_ONLY}" != "ON" ]; then
    install_custom_pkg
fi

# ---- 编译 ----
BUILD_DIR="${SCRIPT_DIR}/build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release)
if [ "${GEIR_BUILD_ONLY}" = "ON" ]; then
    CMAKE_FLAGS+=(-DBUILD_EAGER=OFF -DBUILD_GEIR=ON)
elif [ "${EAGER_ONLY}" = "ON" ]; then
    CMAKE_FLAGS+=(-DBUILD_EAGER=ON -DBUILD_GEIR=OFF)
else
    CMAKE_FLAGS+=(-DBUILD_EAGER=ON -DBUILD_GEIR=ON)
fi

echo "[run.sh] cmake ${CMAKE_FLAGS[*]} ..."
cmake "${CMAKE_FLAGS[@]}" -S "${SCRIPT_DIR}" -B "${BUILD_DIR}"
echo "[run.sh] build ..."
cmake --build "${BUILD_DIR}" -j"$(nproc)"

# 仅编译 geir：到此结束（编译验收）。
if [ "${GEIR_BUILD_ONLY}" = "ON" ]; then
    echo "[run.sh] geir example compiled (build-only): ${BUILD_DIR}/test_geir_bitwise_not"
    exit 0
fi

# ---- 运行 eager 示例（真实 NPU，两段式 aclnnBitwiseNot）----
echo "[run.sh] run eager example from CWD: ${RUN_DIR}"
( cd "${RUN_DIR}" && "${BUILD_DIR}/test_aclnn_bitwise_not" )
EAGER_RC=$?
echo "[run.sh] eager example exit=${EAGER_RC}"

# ---- 可选：实跑 geir 示例（需完整 GE 图执行环境）----
if [ "${EAGER_ONLY}" != "ON" ] && [ "${RUN_GEIR}" = "ON" ]; then
    echo "[run.sh] run geir example from CWD: ${RUN_DIR}"
    ( cd "${RUN_DIR}" && "${BUILD_DIR}/test_geir_bitwise_not" ) || echo "[run.sh] geir run returned non-zero (需完整 GE 图执行环境，编译已通过)"
fi

exit ${EAGER_RC}
