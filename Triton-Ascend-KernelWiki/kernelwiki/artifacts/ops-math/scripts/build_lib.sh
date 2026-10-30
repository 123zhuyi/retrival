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
# Library, binary, and package build functions.

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} -UENABLE_STATIC ..

  for lib in "${BUILD_LIBS[@]}"; do
    echo "Building target ${lib}"
    cmake --build . --target ${lib} -- ${VERBOSE} -j $THREAD_NUM
  done

  echo $dotted_line
  echo "Build libs ${BUILD_LIBS[@]} success"
  echo $dotted_line
}

build_binary() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi

  echo $dotted_line
  echo "Start to build binary"

  echo "--------------- prepare build start ---------------"
  local all_targets=$(cmake --build . --target help)
  if grep -wq "gen_bin_scripts" <<< "${all_targets}"; then
    echo "[INFO] Begin to execute build target: gen_bin_scripts."
    cmake --build . --target gen_bin_scripts -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then
      echo "[ERROR] Failed to execute gen_bin_scripts."
      exit 1;
    fi
  else
    echo "[WARNING] Build target 'gen_bin_scripts' not found in cmake targets, available targets: ${all_targets}"
  fi
  echo "--------------- prepare build end ---------------"

  echo "--------------- binary build start ---------------"
  local cur_path=$(pwd)
  if [ "$ENABLE_CUSTOM" == "TRUE" ]; then
    cmake --build . --target ophost_math -- ${VERBOSE} -j $THREAD_NUM
  fi
  if [ ! -L op_impl/ai_core/tbe/op_tiling/liboptiling.so ]; then
    mkdir -p ${cur_path}/op_impl/ai_core/tbe/op_tiling
    ln -s ${cur_path}/libophost_math.so op_impl/ai_core/tbe/op_tiling/liboptiling.so
  fi
  export ASCEND_CUSTOM_OPP_PATH=${cur_path}

  local UNITS=(${COMPUTE_UNIT_SHORT//;/ })
  if [[ ${#UNITS[@]} -eq 0 ]]; then
    UNITS+=("ascend910b")
  fi
  for unit in "${UNITS[@]}"; do
    if grep -wq "prepare_binary_compile_${unit}"<<<"${all_targets}"; then
      echo "[INFO] Begin to prepare binary compile for target: ${unit}"
      cmake --build . --target prepare_binary_compile_${unit} -- ${VERBOSE} -j $THREAD_NUM
      if [ $? -ne 0 ]; then
        echo "[ERROR] Kernel compile failed!" && exit 1
      fi
      local opc_list_num=$(wc -l <"${BUILD_PATH}/binary/${unit}/bin/opc_cmd/opc_cmd.sh")
      CMAKE_ARGS="${CMAKE_ARGS} -DOPC_NUM_${unit}=${opc_list_num}"
    fi
  done

  echo "[INFO] CMAKE_ARGS is: ${CMAKE_ARGS}"
  cd "$BUILD_PATH" && cmake .. ${CMAKE_ARGS}

  if grep -wq "binary" <<< "${all_targets}"; then
    echo "[INFO] Start to compile kernel binary."
    cmake --build . --target binary -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then
      echo "[ERROR] Kernel compile failed!" && exit 1
    fi
  else
    echo "[WARNING] Compile kernel binary failed! Build target 'binary' not found in cmake targets. Available targets: ${all_targets}"
  fi
  if grep -wq "gen_bin_info_config" <<< "${all_targets}"; then
    cmake --build . --target gen_bin_info_config -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  else
    echo "[WARNING] Generate binary info config failed! Build target 'gen_bin_info_config' not found in cmake targets. Available targets: ${all_targets}"
  fi
  echo "--------------- binary build end ---------------"

  echo "Build binary success"
  echo $dotted_line
}

build_static_lib() {
  echo $dotted_line
  echo "Start to build static lib."

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  local all_targets=$(cmake --build . --target help)
  rm -fr ${BUILD_PATH}/bin_tmp
  mkdir -p ${BUILD_PATH}/bin_tmp
  if grep -wq "ophost_math_static" <<<"${all_targets}"; then
    cmake --build . --target ophost_math_static -- ${VERBOSE} -j $THREAD_NUM
  fi

  local UNITS=(${COMPUTE_UNIT_SHORT//;/ })
  if [[ ${#UNITS[@]} -eq 0 ]]; then
    UNITS+=("ascend910b")
  fi
  cmake --build . --target opapi_math_static -- ${VERBOSE} -j $THREAD_NUM
  local jit_command=""
  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    jit_command="-j"
  fi
  for unit in "${UNITS[@]}"; do
    rm -fr ${BUILD_PATH}/bin_tmp/${unit}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" GenStaticOpResourceIni -s ${unit} -b ${BUILD_PATH} ${jit_command}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" StaticCompile -s ${unit} -b ${BUILD_PATH} -n=0 -a=${ARCH_INFO} ${jit_command}
  done
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  cmake --build . --target cann_math_static -- ${VERBOSE} -j $THREAD_NUM
  echo "Build static lib success!"
}

build_package() {
  echo "--------------- build package start ---------------"
  clean_build_out

  local all_targets=$(cmake --build . --target help)
  if [[ "$ENABLE_BINARY" != "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    if grep -wq "ascendc_impl_gen" <<< "${all_targets}"; then
      cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
      if [ $? -ne 0 ]; then exit 1; fi
    fi
  fi

  if grep -wq "build_es_math" <<< "${all_targets}"; then
    cmake --build . --target build_es_math -- ${VERBOSE} -j $THREAD_NUM
    [ $? -ne 0 ] && echo "[ERROR] target:build_es_math compile failed!" && exit 1
  fi
  cmake --build . --target package -- ${VERBOSE} -j $THREAD_NUM
  echo "--------------- build package end ---------------"
}

build_package_static() {
  if [ ! -d "$BUILD_OUT_PATH" ]; then
    echo "Error: Directory $BUILD_OUT_PATH does not exist"
    return 1
  fi

  local run_files=("$BUILD_OUT_PATH"/*.run)
  if [ ${#run_files[@]} -eq 0 ]; then
    echo "Error: No .run files found in $BUILD_OUT_PATH directory"
    return 1
  fi
  if [ ${#run_files[@]} -gt 1 ]; then
    echo "Error: Multiple .run files found in $BUILD_OUT_PATH directory:"
    printf '%s\n' "${run_files[@]}"
    return 1
  fi

  local run_file=$(basename "${run_files[0]}")
  echo "Found .run file: $run_file"
  if [[ "$run_file" != *"ops-math"* ]]; then
    echo "Error: Filename '$run_file' does not contain 'ops-math'"
    return 1
  fi
  local new_name="${run_file/ops-math/ops-math-static}"
  new_name="${new_name%.run}"

  local static_files_dir="$BUILD_PATH/static_library_files"
  if [ ! -d "$static_files_dir" ]; then
    echo "Error: Directory $static_files_dir does not exist"
    return 1
  fi
  if [ -z "$(ls -A "$static_files_dir")" ]; then
    echo "Error: Directory $static_files_dir is empty"
    return 1
  fi

  local new_dir_path="$BUILD_PATH/$new_name"
  if mv "$static_files_dir" "$new_dir_path"; then
    echo "Preparing for packaging: renamed $static_files_dir to $new_dir_path"
  else
    echo "Packaging preparation failed: directory rename failed ($static_files_dir -> $new_dir_path)"
    return 1
  fi

  local new_filename="${new_name}.tar.gz"
  if tar -czf "$BUILD_OUT_PATH/$new_filename" -C "$BUILD_PATH" "$new_name"; then
    echo "[SUCCESS] Build static lib success!"
    echo "Successfully created compressed package: $BUILD_OUT_PATH/$new_filename"
    echo "Restoring original directory name: $new_dir_path -> $static_files_dir"
    mv "$new_dir_path" "$static_files_dir"
    return 0
  else
    echo "Error: Failed to create compressed package"
    mv "$new_dir_path" "$static_files_dir"
    return 1
  fi
}
