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
# Global config: variables, paths, utility functions shared across all build scripts.

RELEASE_TARGETS=("ophost" "opapi" "opgraph" "opkernel" "opkernel_aicpu" "onnxplugin" "tfplugin")
SUPPORT_COMPUTE_UNIT_SHORT=("ascend910b" "ascend910_93" "ascend350" "ascend950" "ascend310p" "ascend910" "ascend310b" "ascend630" "ascend610lite" "ascend031" "ascend035" "kirinx90" "kirin9030" "mc62")
SUPPORTED_SHORT_OPTS="hj:vO:uf:-:"

SUPPORTED_LONG_OPTS=(
  "help" "ops=" "soc=" "vendor_name=" "debug" "cov" "noexec" "aicpu" "noaicpu" "opkernel" "opkernel_aicpu" "jit"
  "pkg" "asan" "valgrind" "make_clean" "static" "build-type=" "no_force" "simulator"
  "ophost" "opapi" "opgraph"
  "run_example" "genop=" "genop_aicpu=" "experimental" "cann_3rd_lib_path" "mssanitizer" "oom" "onnxplugin" "tfplugin"
  "dump_cce" "bisheng_flags=" "kernel_template_input=" "module_extension=" "example_name=" "rule_launch=" "gtest_filter=" "ccache="
  "pkg-type="
)

dotted_line="----------------------------------------------------------------"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
ARCH_INFO=$(uname -m)
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/compiler/include"
export GRAPH_INCLUDE_PATH="${INCLUDE_PATH}/graph"
export GE_INCLUDE_PATH="${INCLUDE_PATH}/ge"
export GE_EXTERNAL_INCLUDE_PATH="${INCLUDE_PATH}/external"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"

USER_SET_SLOG=${ASCEND_SLOG_PRINT_TO_STDOUT:+true}
USER_SET_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:+true}
if [[ ! -v ASCEND_SLOG_PRINT_TO_STDOUT ]]; then
  export ASCEND_SLOG_PRINT_TO_STDOUT=1
fi
if [[ ! -v ASCEND_GLOBAL_LOG_LEVEL ]]; then
  export ASCEND_GLOBAL_LOG_LEVEL=1
fi

in_array() {
  local needle="$1"
  shift
  local haystack=("$@")
  for item in "${haystack[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

check_option_validity() {
  local arg="$1"

  if [[ "$arg" =~ ^-[^-] ]]; then
    local opt_chars=${arg:1}
    local needs_arg_opts=$(echo "$SUPPORTED_SHORT_OPTS" | grep -o "[a-zA-Z]:" | tr -d ':')
    local i=0
    while [ $i -lt ${#opt_chars} ]; do
      local char="${opt_chars:$i:1}"
      if [[ ! "$SUPPORTED_SHORT_OPTS" =~ "$char" ]]; then
        echo "[ERROR] Invalid short option: -$char"
        return 1
      fi
      if [[ "$needs_arg_opts" =~ "$char" ]]; then
        while [ $i -lt ${#opt_chars} ] && [[ "${opt_chars:$i:1}" =~ [0-9a-zA-Z] ]]; do
          i=$((i + 1))
        done
      else
        i=$((i + 1))
      fi
    done
    return 0
  fi

  if [[ "$arg" =~ ^-- ]]; then
    local long_opt="${arg:2}"
    local opt_name="${long_opt%%=*}"
    for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
      if [[ "$supported_opt" =~ =$ ]]; then
        local base_opt="${supported_opt%=}"
        if [[ "$opt_name" == "$base_opt" ]]; then
          return 0
        fi
      else
        if [[ "$opt_name" == "$supported_opt" ]]; then
          return 0
        fi
      fi
    done
    echo "[ERROR] Invalid long option: --$opt_name"
    return 1
  fi

  return 0
}

check_pkg_type() {
  local pkg_type="$1"
  if [[ "$pkg_type" != "run" && "$pkg_type" != "rpm" && "$pkg_type" != "deb" && "$pkg_type" != "all" ]]; then
    echo "[ERROR] --pkg-type only supports run/rpm/deb/all, got: $pkg_type"
    exit 1
  fi
}
