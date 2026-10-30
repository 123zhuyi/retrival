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
VENDOR_NAME=${VENDOR_NAME:-custom}
REPO_FAMILY=${REPO_FAMILY:-math}
DEVICE_ID=${DEVICE_ID:-0}

case "$(uname -m)" in
    aarch64|arm64)
        HOST_ARCH_DIR="aarch64-linux"
        ;;
    x86_64|amd64)
        HOST_ARCH_DIR="x86_64-linux"
        ;;
    *)
        echo "Unsupported host architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

CUSTOM_OP_API_DIR="${CANN_ROOT}/opp/vendors/${VENDOR_NAME}_${REPO_FAMILY}/op_api"
OPS_REPO_ROOT="${OPS_REPO:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
LOCAL_BUILD_LIB_DIR="${OPS_REPO_ROOT}/build"

source "${CANN_ROOT}/set_env.sh"
export LD_LIBRARY_PATH="${LOCAL_BUILD_LIB_DIR}:${CUSTOM_OP_API_DIR}/lib:${LD_LIBRARY_PATH:-}"
export ACL_DEVICE_ID="${DEVICE_ID}"

BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"
PACKAGE_INCLUDE_DIR="${OPS_REPO_ROOT}/build_out/_CPack_Packages/Linux/External/cann-ops-${REPO_FAMILY}-${VENDOR_NAME}_linux-${HOST_ARCH_DIR%%-*}.run/packages/vendors/${VENDOR_NAME}_${REPO_FAMILY}/op_api/include"

g++ -std=c++17 -O2 \
    "${SCRIPT_DIR}/test_aclnn_sqrt_grad.cpp" \
    -I"${CANN_ROOT}/${HOST_ARCH_DIR}/include" \
    -I"${CANN_ROOT}/${HOST_ARCH_DIR}/include/aclnnop" \
    -I"${PACKAGE_INCLUDE_DIR}" \
    -I"${CUSTOM_OP_API_DIR}/include" \
    -I"${LOCAL_BUILD_LIB_DIR}" \
    -L"${LOCAL_BUILD_LIB_DIR}" \
    -L"${CANN_ROOT}/${HOST_ARCH_DIR}/lib64" \
    -L"${CUSTOM_OP_API_DIR}/lib" \
    -Wl,-rpath,"${LOCAL_BUILD_LIB_DIR}:${CANN_ROOT}/${HOST_ARCH_DIR}/lib64:${CUSTOM_OP_API_DIR}/lib" \
    -lcust_opapi -lnnopbase -lascendcl \
    -o "${BUILD_DIR}/test_aclnn_sqrt_grad"

"${BUILD_DIR}/test_aclnn_sqrt_grad"
