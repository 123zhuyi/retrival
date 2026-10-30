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
# CMake argument assembly and project initialization.

custom_cmake_args() {
  if [[ -n $COMPILED_OPS ]]; then
    COMPILED_OPS="${COMPILED_OPS//,/;}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_OP_NAME=$COMPILED_OPS"
  fi
  if [[ -n $VENDOR_NAME ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DVENDOR_NAME=$VENDOR_NAME"
  fi
}

assemble_cmake_args() {
  if [[ "$ENABLE_ASAN" == "TRUE" ]]; then
    set +e
    echo 'int main() {return 0;}' | gcc -x c -fsanitize=address - -o asan_test >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "This environment does not have the ASAN library, no need enable ASAN"
      ENABLE_ASAN=FALSE
    else
      $(rm -f asan_test)
      CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ASAN=TRUE"
    fi
    set -e
  fi
  if [[ "$ENABLE_VALGRIND" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_VALGRIND=TRUE"
  fi
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TEST=TRUE"
    custom_cmake_args
  fi
  if [[ "$ENABLE_UT_EXEC" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_UT_EXEC=TRUE"
  fi
  if [[ "$ENABLE_COVERAGE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_COVERAGE=TRUE"
  fi
  if [[ "$ENABLE_BINARY" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_BINARY=TRUE"
  fi
  if [[ "$ENABLE_CUSTOM" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CUSTOM=TRUE -DENABLE_BINARY=TRUE"
    custom_cmake_args
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=TRUE"
  fi
  if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_EXPERIMENTAL=TRUE"
  fi
  if [[ "x$BUILD_MODE" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DBUILD_MODE=${BUILD_MODE}"
  fi
  if [[ "x$BISHENG_FLAGS" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DBISHENG_FLAGS=${BISHENG_FLAGS}"
  fi
  if [[ "x$KERNEL_TEMPLATE_INPUT" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DKERNEL_TEMPLATE_INPUT=${KERNEL_TEMPLATE_INPUT}"
  fi
  if [[ "$OP_HOST_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_HOST_UT=TRUE"
  fi
  if [[ "$OP_API_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_API_UT=TRUE"
  fi
  if [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_GRAPH_UT=TRUE"
  fi
  if [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_KERNEL_UT=TRUE"
  fi
  if [[ "$OP_KERNEL_AICPU_UT" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DOP_KERNEL_AICPU_UT=TRUE"
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST_ALL=TRUE"
  fi
  if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_STATIC=${ENABLE_STATIC}"
  fi
  if [[ -n "${BUILD_TYPE}" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  fi
  if [[ "$ENABLE_MSSANITIZER" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_MSSANITIZER=TRUE"
  fi
  if [[ "$ENABLE_OOM" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_OOM=TRUE"
  fi
  if [[ "$DISABLE_AICPU" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DDISABLE_AICPU=TRUE"
  fi
  if [[ "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_DUMP_CCE=TRUE"
  fi
  if [[ "$NO_FORCE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DNO_FORCE=TRUE"
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CCACHE=${ENABLE_CCACHE}"
  if [[ "x$ENABLE_RULE_LAUNCH" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DRULE_LAUNCH=${ENABLE_RULE_LAUNCH}"
  fi
  if [[ -n $GTEST_FILTER ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGTEST_FILTER=${GTEST_FILTER}"
  fi
  if [[ -n $COMPUTE_UNIT ]]; then
    COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | tr '[:upper:]' '[:lower:]')
    found=0
    best_match=""
    best_match_len=0
    for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
      if [[ "$COMPUTE_UNIT" == "$support_unit" ]]; then
        COMPUTE_UNIT_SHORT=$support_unit
        found=1
        break
      fi
      if [[ "$support_unit" == "$COMPUTE_UNIT"* ]] || [[ "$COMPUTE_UNIT" == "$support_unit"* ]]; then
        local match_len=${#support_unit}
        if [[ $match_len -gt $best_match_len ]]; then
          best_match=$support_unit
          best_match_len=$match_len
        fi
      fi
    done
    if [[ $found -eq 0 ]]; then
      if [[ -n "$best_match" ]]; then
        COMPUTE_UNIT_SHORT=$best_match
        COMPUTE_UNIT=$best_match
        found=1
      else
        echo "soc only support : ${SUPPORT_COMPUTE_UNIT_SHORT[@]}"
        exit 1
      fi
    fi
    echo "COMPUTE_UNIT: ${COMPUTE_UNIT}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT"
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
  CMAKE_ARGS="$CMAKE_ARGS -DMODULE_EXT=${MODULE_EXT}"
}

cmake_init() {
  if [[ "$ENABLE_GENOP" == "TRUE" || "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
    return
  fi
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi
  if [ ! -d "${BUILD_OUT_PATH}" ]; then
    mkdir -p "${BUILD_OUT_PATH}"
  fi

  [ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f ${BUILD_PATH}/CMakeCache.txt

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
}
