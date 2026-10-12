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
# Example build and run functions.

get_simulator_chip_version() {
  local soc=$1
  case "$soc" in
    ascend910) echo "dav_1001" ;;
    ascend910_93|ascend910b) echo "dav_2201" ;;
    ascend310p) echo "dav_2002" ;;
    ascend310b) echo "dav_3002" ;;
    ascend950) echo "dav_3510" ;;
    ascend350) echo "dav_3510" ;;
    *)
      echo "[ERROR] Unsupported soc version for simulator: $soc" >&2
      return 1
      ;;
  esac
}

get_simulator_args() {
  if [[ "$ENABLE_SIMULATOR" == "FALSE" ]];then
    return 0
  fi
  if [[ "$ENABLE_SIMULATOR" == "TRUE" ]] && [[ -n "$COMPUTE_UNIT" ]]; then
    local chip_version=$(get_simulator_chip_version "$COMPUTE_UNIT")
    if [[ $? -ne 0 ]]; then
      exit 1
    fi
    if [[ -n "$chip_version" ]]; then
      local sim_lib_path="${ASCEND_HOME_PATH}/tools/simulator/${chip_version}/lib"
      if [[ ! -d "$sim_lib_path" ]]; then
        echo "[ERROR] Simulator lib path not found: $sim_lib_path" >&2
        exit 1
      else
        echo "[INFO] Successfully linked simulator libraries: ${sim_lib_path}/libruntime_camodel.so, ${sim_lib_path}/libnpu_drv_camodel.so" >&2
      fi
      echo "$sim_lib_path"
      return 0
    fi
  fi
  echo ""
  return 1
}

compile_eager_example_default() {
  local source_file=$1
  local executable_name=$2
  local sim_lib_path=$3

  if [[ -n "$sim_lib_path" ]]; then
    if [[ "$USER_SET_SLOG" != "true" ]]; then
      export ASCEND_SLOG_PRINT_TO_STDOUT=0
    fi
    if [[ "$USER_SET_LOG_LEVEL" != "true" ]]; then
      export ASCEND_GLOBAL_LOG_LEVEL=3
    fi
    export LD_LIBRARY_PATH=${sim_lib_path}:${LD_LIBRARY_PATH}
    ln -sf ${sim_lib_path}/libruntime_camodel.so ${sim_lib_path}/libruntime.so
    ln -sf ${sim_lib_path}/libnpu_drv_camodel.so ${sim_lib_path}/libascend_hal.so
    g++ ${source_file} \
      -I ${INCLUDE_PATH} \
      -I ${ACLNN_INCLUDE_PATH} \
      -L ${EAGER_LIBRARY_PATH} \
      -lopapi_math -lascendcl -lnnopbase \
      -L ${sim_lib_path} \
      -lruntime_camodel -lnpu_drv_camodel \
      -o ${executable_name} \
      -Wl,-rpath=${sim_lib_path}
  else
    g++ ${source_file} \
      -I ${INCLUDE_PATH} \
      -I ${ACLNN_INCLUDE_PATH} \
      -L ${EAGER_LIBRARY_PATH} \
      -lopapi_math -lascendcl -lnnopbase \
      -o ${executable_name}
  fi
}

compile_eager_example_cust() {
  local source_file=$1
  local executable_name=$2
  local sim_lib_path=$3

  echo "pkg_mode:${PKG_MODE} vendor_name:${VENDOR}"

  local cust_include_flags=""
  local cust_library_flags=""
  local cust_rpath_flags=""
  local cust_aclnnop_paths=""

  if [[ -n "${ASCEND_CUSTOM_OPP_PATH}" ]]; then
    IFS=':' read -ra PATH_ARRAY <<< "${ASCEND_CUSTOM_OPP_PATH}"
    for path in "${PATH_ARRAY[@]}"; do
      cust_include_flags="${cust_include_flags} -I ${path}/op_api/include"
      cust_library_flags="${cust_library_flags} -L ${path}/op_api/lib"
      cust_rpath_flags="${cust_rpath_flags}:${path}/op_api/lib"
      cust_aclnnop_paths="${cust_aclnnop_paths} ${path}/op_api/include/aclnnop"
    done
    cust_rpath_flags="${cust_rpath_flags#:}"
  else
    cust_include_flags="-I ${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_math/op_api/include"
    cust_library_flags="-L ${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_math/op_api/lib"
    cust_rpath_flags="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_math/op_api/lib"
    cust_aclnnop_paths="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR}_math/op_api/include/aclnnop"
  fi

  local include_dir_mode=""
  for aclnnop_path in ${cust_aclnnop_paths}; do
    local include_dir=$(dirname ${aclnnop_path})
    if [[ -z "${include_dir_mode}" ]]; then
      include_dir_mode=$(stat -c %a ${include_dir} 2>/dev/null)
    fi
    if [ ! -L ${aclnnop_path} ]; then
      chmod u+w ${include_dir} 2>/dev/null
      ln -s ${include_dir} ${aclnnop_path} 2>/dev/null
    fi
  done

  if [[ -n "$sim_lib_path" ]]; then
    if [[ "$USER_SET_SLOG" != "true" ]]; then
      export ASCEND_SLOG_PRINT_TO_STDOUT=0
    fi
    if [[ "$USER_SET_LOG_LEVEL" != "true" ]]; then
      export ASCEND_GLOBAL_LOG_LEVEL=3
    fi
    export LD_LIBRARY_PATH=${sim_lib_path}:${LD_LIBRARY_PATH}
    ln -sf ${sim_lib_path}/libruntime_camodel.so ${sim_lib_path}/libruntime.so
    ln -sf ${sim_lib_path}/libnpu_drv_camodel.so ${sim_lib_path}/libascend_hal.so
    g++ ${source_file} \
      ${cust_include_flags} \
      -I ${INCLUDE_PATH} \
      -I ${INCLUDE_PATH}/aclnnop \
      ${cust_library_flags} \
      -L ${EAGER_LIBRARY_PATH} \
      -lcust_opapi -lascendcl -lnnopbase \
      -L ${sim_lib_path} \
      -lruntime_camodel -lnpu_drv_camodel \
      -o ${executable_name} \
      -Wl,-rpath=${cust_rpath_flags}:${sim_lib_path}
  else
    g++ ${source_file} \
      ${cust_include_flags} \
      -I ${INCLUDE_PATH} \
      -I ${INCLUDE_PATH}/aclnnop \
      ${cust_library_flags} \
      -L ${EAGER_LIBRARY_PATH} \
      -lcust_opapi -lascendcl -lnnopbase \
      -o ${executable_name} \
      -Wl,-rpath=${cust_rpath_flags}
  fi

  for aclnnop_path in ${cust_aclnnop_paths}; do
    if [ -L ${aclnnop_path} ]; then
      local include_dir=$(dirname ${aclnnop_path})
      rm ${aclnnop_path} 2>/dev/null
      chmod ${include_dir_mode} ${include_dir} 2>/dev/null
    fi
  done
}

compile_eager_example() {
  local source_file=$1
  local executable_name=$2
  local sim_lib_path
  sim_lib_path=$(get_simulator_args)
  local ret=$?
  if [ $ret -ne 0 ]; then
    exit 1
  fi

  if [[ "${PKG_MODE}" == "" ]]; then
    compile_eager_example_default "${source_file}" "${executable_name}" "${sim_lib_path}"
  elif [[ "${PKG_MODE}" == "cust" ]]; then
    compile_eager_example_cust "${source_file}" "${executable_name}" "${sim_lib_path}"
  else
    echo "Error: pkg_mode(${PKG_MODE}) must be cust or empty."
    exit 1
  fi
}

compile_graph_example() {
  local source_file=$1
  local executable_name=$2

  g++ ${source_file} \
    -I ${GE_EXTERNAL_INCLUDE_PATH} \
    -I ${GRAPH_INCLUDE_PATH} \
    -I ${GE_INCLUDE_PATH} \
    -I ${INCLUDE_PATH} \
    -I ${INC_INCLUDE_PATH} \
    -L ${GRAPH_LIBRARY_PATH} \
    -lgraph -lge_runner -lgraph_base -lge_compiler \
    -o ${executable_name}
}

build_example() {
  echo $dotted_line
  echo "Start to run examples, name:${EXAMPLE_NAME} mode:${EXAMPLE_MODE}"

  local file_pattern=""
  local search_path=""
  local executable_name=""

  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    executable_name="test_aclnn_${EXAMPLE_NAME}"
    if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
      search_path="../experimental"
    else
      search_path="../"
    fi
    file_pattern="test_aclnn_*.cpp"
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    executable_name="test_geir_${EXAMPLE_NAME}"
    if [[ "$ENABLE_EXPERIMENTAL" == "TRUE" ]]; then
      search_path="../experimental"
    else
      search_path="../"
    fi
    file_pattern="test_geir_*.cpp"
  else
    usage
    exit 1
  fi

  local find_cmd="find ${search_path}"
  if [[ "$COMPUTE_UNIT" == "ascend950" ]]; then
    find_cmd="${find_cmd} \\( -path \"*/${EXAMPLE_NAME}/examples/*\" -o -path \"*/${EXAMPLE_NAME}/examples/arch35/*\" \\)"
  else
    find_cmd="${find_cmd} -path \"*/${EXAMPLE_NAME}/examples/*\""
  fi
  find_cmd="${find_cmd} -name \"${file_pattern}\" -not -path \"*/scripts/*\""
  if [[ "$ENABLE_EXPERIMENTAL" != "TRUE" ]]; then
    find_cmd="${find_cmd} -not -path \"*/experimental/*\""
  fi

  local files=$(eval $find_cmd)

  if [ -z "$files" ]; then
    echo "ERROR: ${EXAMPLE_NAME} does not have ${EXAMPLE_MODE} examples"
    exit 1
  fi

  for f in $files; do
    if [[ -n "$SINGLE_EXAMPLE" ]]; then
      local example=$(basename "$f" .cpp)
      example=${example#test_aclnn_}
      example=${example#test_geir_}
      if [[ "$example" != "$SINGLE_EXAMPLE" ]]; then
        echo "Skip $f (--example_name=$SINGLE_EXAMPLE specified)"
        continue
      fi
    fi

    echo "Start compile and run examples file: $f"

    if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
      compile_eager_example "$f" "$executable_name"
    elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
      compile_graph_example "$f" "$executable_name"
    fi

    if [[ "$ENABLE_UT_EXEC" == "TRUE" ]]; then
      ./${executable_name}
      local exit_code=$?

      if [ $exit_code -eq 0 ]; then
        echo "run ${executable_name}, execute samples success"
      else
        echo "run ${executable_name}, execute samples failed"
        exit 1
      fi
    else
      echo "Skip running ${executable_name} (--noexec specified, binary is ready for cannsim)"
    fi
  done
}
