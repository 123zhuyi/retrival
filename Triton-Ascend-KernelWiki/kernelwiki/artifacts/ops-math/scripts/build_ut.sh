#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================
#
# Unit test build functions.

build_ut() {
  echo $dotted_line
  echo "Start to build ut"

  for lib in "${UT_TARGETS[@]}"; do
    $(find . -name "${lib}*" -type f -delete)
    cmake --build . --target ${lib} -- ${VERBOSE} -j $THREAD_NUM
  done
  if [[ "$ENABLE_COVERAGE" =~ "TRUE" ]]; then
    cmake --build . --target generate_ops_cpp_cov -- -j $THREAD_NUM
  fi
}
