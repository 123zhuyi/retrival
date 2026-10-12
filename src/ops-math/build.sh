#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export BASE_PATH="${SCRIPT_DIR}"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
REPOSITORY_NAME="math"

source "${SCRIPT_DIR}/scripts/build.conf.sh"
source "${SCRIPT_DIR}/scripts/build_clean.sh"
source "${SCRIPT_DIR}/scripts/build_options.sh"
source "${SCRIPT_DIR}/scripts/build_cmake.sh"
source "${SCRIPT_DIR}/scripts/build_lib.sh"
source "${SCRIPT_DIR}/scripts/build_ut.sh"
source "${SCRIPT_DIR}/scripts/build_example.sh"
source "${SCRIPT_DIR}/scripts/build_genop.sh"

main() {
  checkopts "$@"
  assemble_cmake_args
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"

  clean_build_binary
  cmake_init
  if [ "$ENABLE_CREATE_LIB" == "TRUE" ]; then
    build_lib
  fi
  if [[ "$ENABLE_BINARY" == "TRUE" || "$ENABLE_CUSTOM" == "TRUE" ]] && [[ "$ENABLE_JIT" == "FALSE" ]]; then
    build_binary
  fi
  if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    build_static_lib
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    build_package
    if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
      build_package_static
    fi
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    build_ut
  fi
  if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
    build_example
  fi
  if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
    gen_op
  fi
  if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
    gen_aicpu_op
  fi
}

if [ $# -eq 0 ]; then
  usage
  exit 0
fi
main "$@" 2>&1 | while IFS= read -r line; do echo "$(date '+[%Y-%m-%d %H:%M:%S]') $line"; done
