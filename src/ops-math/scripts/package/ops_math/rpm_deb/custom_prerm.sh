#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

sourcedir="${INSTALL_PATH}"
WHL_INSTALL_DIR_PATH="${sourcedir}/python/site-packages"
export PYTHONPATH="${WHL_INSTALL_DIR_PATH}"
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip() { python3 -m pip "$@" || pip3 "$@"; }
run_pip uninstall -y es-math >/dev/null 2>&1 || true

rm -rf "${WHL_INSTALL_DIR_PATH}/es_math" 2>/dev/null
rm -rf "${WHL_INSTALL_DIR_PATH}/es_math-"*.dist-info 2>/dev/null

rm -f "${sourcedir}/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/__init__.py"
rm -f "${sourcedir}/opp/built-in/op_impl/ai_core/tbe/impl/ops_math/dynamic/__init__.py"

[ -d "${WHL_INSTALL_DIR_PATH}" ] && rmdir "${WHL_INSTALL_DIR_PATH}" 2>/dev/null || true
parent=$(dirname "${WHL_INSTALL_DIR_PATH}")
[ -d "${parent}" ] && rmdir "${parent}" 2>/dev/null || true

rm -f "${sourcedir}"/ops_math/es_packages/whl/*.whl 2>/dev/null || true
[ -d "${sourcedir}"/ops_math/es_packages/whl ] && rmdir "${sourcedir}"/ops_math/es_packages/whl 2>/dev/null || true
[ -d "${sourcedir}"/ops_math/es_packages ] && rmdir "${sourcedir}"/ops_math/es_packages 2>/dev/null || true
[ -d "${sourcedir}"/ops_math ] && rmdir "${sourcedir}"/ops_math 2>/dev/null || true
