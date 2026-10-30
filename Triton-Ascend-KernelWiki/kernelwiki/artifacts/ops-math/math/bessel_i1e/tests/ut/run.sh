#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "========================================"
echo "BesselI1e UT Tests"
echo "========================================"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Configuring..."
cmake .. > /dev/null

echo "Building..."
make -j$(nproc) > /dev/null

echo "Running..."
echo ""
./test_bessel_i1e_ut
RESULT=$?

echo ""
if [ $RESULT -eq 0 ]; then
    echo "========================================"
    echo "UT Result: PASS"
    echo "========================================"
else
    echo "========================================"
    echo "UT Result: FAIL"
    echo "========================================"
fi

exit $RESULT
