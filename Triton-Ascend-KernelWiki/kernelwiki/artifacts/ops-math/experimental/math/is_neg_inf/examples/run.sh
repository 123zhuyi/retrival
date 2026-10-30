#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CANN_ROOT=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}

case "$(uname -m)" in
    aarch64|arm64)
        HOST_ARCH_DIR="aarch64-linux"
        ;;
    x86_64|amd64)
        HOST_ARCH_DIR="x86_64-linux"
        ;;
    *)
        echo "unsupported host arch: $(uname -m)" >&2
        exit 1
        ;;
esac

source "${CANN_ROOT}/set_env.sh"
CUSTOM_OPP_ROOT="${ASCEND_CUSTOM_OPP_PATH-}"
CUSTOM_OPP_ROOT="${CUSTOM_OPP_ROOT%%:*}"
if [[ -z "${CUSTOM_OPP_ROOT}" ]]; then
    CUSTOM_OPP_ROOT="${CANN_ROOT}/opp/vendors/custom_math"
fi
export LD_LIBRARY_PATH="${CUSTOM_OPP_ROOT}/op_api/lib:${LD_LIBRARY_PATH:-}"

BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

g++ -std=c++17 -O2 \
    "${SCRIPT_DIR}/test_aclnn_is_neg_inf.cpp" \
    -I"${CANN_ROOT}/${HOST_ARCH_DIR}/include" \
    -I"${CANN_ROOT}/${HOST_ARCH_DIR}/include/aclnnop" \
    -I"${CUSTOM_OPP_ROOT}/op_api/include" \
    -L"${CANN_ROOT}/${HOST_ARCH_DIR}/lib64" \
    -L"${CUSTOM_OPP_ROOT}/op_api/lib" \
    -Wl,-rpath,"${CANN_ROOT}/${HOST_ARCH_DIR}/lib64:${CUSTOM_OPP_ROOT}/op_api/lib" \
    -lcust_opapi -lnnopbase -lascendcl \
    -o "${BUILD_DIR}/test_aclnn_is_neg_inf"

"${BUILD_DIR}/test_aclnn_is_neg_inf"
