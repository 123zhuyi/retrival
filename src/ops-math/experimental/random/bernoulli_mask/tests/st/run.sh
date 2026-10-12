#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ops_math_root="$(cd "${script_dir}/../../../../.." && pwd)"

if [[ -n "${ASCEND_HOME_PATH:-}" && -r "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_HOME_PATH}/set_env.sh"
elif [[ -r /usr/local/Ascend/cann/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann/set_env.sh
else
    echo "ERROR: CANN set_env.sh was not found; set ASCEND_HOME_PATH." >&2
    exit 1
fi

build_dir="${BERNOULLI_ST_BUILD_DIR:-${ops_math_root}/build_out/bernoulli_mask_st}"
cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j"$(nproc)"

if [[ "${1:-}" != "--noexec" ]]; then
    "${build_dir}/test_aclnn_bernoulli_st"
fi
