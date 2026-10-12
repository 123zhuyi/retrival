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
# Option parsing, validation, and help display.

usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      package)
        echo "Package Build Options:"
        echo $dotted_line
        echo "    --pkg                  Build run package with kernel bin"
        echo "    --pkg-type=<TYPE>      Specify package type(TYPE options: run/rpm/deb/all), Default: run"
        echo "    --static               Build static library package"
        echo "    --jit                  Build run package without kernel bin"
        echo "    --soc=soc_version      Compile for specified Ascend SoC"
        echo "    --vendor_name=name     Specify custom operator package vendor name (cannot be used with --jit)"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple) (cannot be used with --jit)"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --asan                 Enable ASAN (Address Sanitizer) on the host side"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --experimental         Build experimental version"
        echo "    --cann_3rd_lib_path=<PATH>"
        echo "                           Set ascend third_party package install path, default ./third_party"
        echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files"
        echo "    --bisheng_flags=flag1,flag2"
        echo "                           Specify bisheng compiler flags (comma-separated for multiple)"
        echo "    --kernel_template_input='args0=args0;args1=args1'"
        echo "                           Specify kernel template input arguments (semicolon-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
        echo "    bash build.sh --pkg --pkg-type=deb --soc=ascend910b"
        echo "    bash build.sh --pkg --pkg-type=rpm --soc=ascend910b"
        echo "    bash build.sh --pkg --ops=add,sub --build-type=Debug"
        echo "    bash build.sh --pkg --static --soc=ascend910b"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --mssanitizer"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --oom"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --dump_cce"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=abs --bisheng_flags=ccec_g,oom"
        echo "    bash build.sh --pkg --experimental --soc=ascend950 --ops=fills --kernel_template_input='schMode=0;dType=1'"
        return
        ;;
      opkernel)
        echo "Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel             Build binary kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files"
        echo "    --no_force             Don't force dependency installation"
        echo "    --bisheng_flags=flag1,flag2"
        echo "                           Specify bisheng compiler config (comma-separated for multiple)"
        echo "    --kernel_template_input='args0=args0;args1=args1'"
        echo "                           Specify kernel template input arguments (semicolon-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --build-type=Debug"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --mssanitizer"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --oom"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --dump_cce"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=add,sub --bisheng_flags=ccec_g,oom"
        echo "    bash build.sh --opkernel --soc=ascend950 --ops=fills --kernel_template_input='schMode=0;dType=1'"
        return
        ;;
      opkernel_aicpu)
        echo "AICPU Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel_aicpu       Build AICPU kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo "    --mssanitizer          Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --bisheng_flags=flag1,flag2"
        echo "                           Specify bisheng compiler flags (comma-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add,sub"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add,sub --build-type=Debug"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add,sub --mssanitizer"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add,sub --oom"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=add,sub --bisheng_flags=ccec_g,oom"
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -u                     Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --cov                  Enable code coverage for unit tests"
        echo "    --gtest_filter=pattern Run only tests matching the gtest filter pattern"
        echo "    --soc=soc_version      Run unit tests for specified Ascend SoC"
        echo "    --ophost -u            Build and run ophost unit tests"
        echo "    --opapi -u             Build and run opapi unit tests"
        echo "    --opgraph -u           Build and run opgraph unit tests"
        echo "    --opkernel -u          Build and run kernel unit tests"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -u --noexec --cov"
        echo "    bash build.sh -u --ophost --soc=ascend910b --ops=is_finite"
        echo "    bash build.sh --ophost --opapi --opgraph --opkernel -u --cov"
        echo "    bash build.sh -u --ophost --gtest_filter=AddTest*"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    --make_clean           Clean build artifacts"
        echo $dotted_line
        return
        ;;
      valgrind)
        echo "Valgrind Options:"
        echo $dotted_line
        echo "    --valgrind             Run unit tests with valgrind (disables ASAN and noexec)"
        echo $dotted_line
        return
        ;;
      ophost)
        echo "Ophost Build Options:"
        echo $dotted_line
        echo "    --ophost               Build ophost library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --ophost -j16 -O3"
        echo "    bash build.sh --ophost --build-type=Debug"
        return
        ;;
      opapi)
        echo "Opapi Build Options:"
        echo $dotted_line
        echo "    --opapi                Build opapi library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opapi -j16 -O3"
        echo "    bash build.sh --opapi --build-type=Debug"
        return
        ;;
      opgraph)
        echo "Opgraph Build Options:"
        echo $dotted_line
        echo "    --opgraph              Build opgraph library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opgraph -j16 -O3"
        echo "    bash build.sh --opgraph --build-type=Debug"
        return
        ;;
      onnxplugin)
        echo "ONNXPlugin Build Options:"
        echo $dotted_line
        echo "    --onnxplugin           Build onnxplugin library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --onnxplugin -j16 -O3"
        echo "    bash build.sh --onnxplugin --build-type=Debug"
        return
        ;;
      tfplugin)
        echo "TFPlugin Build Options:"
        echo $dotted_line
        echo "    --tfplugin             Build tfplugin library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --ccache=<VALUE>       Enable or disable ccache (VALUE: on/off/true/false/disable), Default: on"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --tfplugin -j16 -O3"
        echo "    bash build.sh --tfplugin --build-type=Debug"
        return
        ;;
      run_example)
        echo "Run examples Options:"
        echo $dotted_line
        echo "    --run_example op_type  mode[eager:graph] [pkg_mode --vendor_name=name --example_name=name]     Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
        echo "    --noexec               Only compile example, do not execute (useful for cross-platform build + cannsim)"
        echo "    --simulator   Enable simulator mode when running aclnn examples"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --run_example abs eager"
        echo "    bash build.sh --run_example abs graph"
        echo "    bash build.sh --run_example abs eager cust"
        echo "    bash build.sh --run_example abs eager cust --vendor_name=custom"
        echo "    bash build.sh --run_example abs eager --simulator --soc=ascend950"
        echo "    bash build.sh --run_example abs eager --example_name=abs --soc=ascend950"
        echo "    bash build.sh --run_example abs eager cust --noexec --vendor_name=custom"
        return
        ;;
      genop)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop=examples/add"
        return
        ;;
      genop_aicpu)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop_aicpu=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop_aicpu=examples/add"
        return
        ;;
    esac
  fi

  echo "build script for ops-math repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-u] "
  echo ""
  echo ""
  echo "Options:"
  echo $dotted_line
  echo "    Build parameters "
  echo $dotted_line
  echo "    -h Print usage"
  echo "    -j[n] Compile thread nums, default is 8, eg: -j8"
  echo "    -v Cmake compile verbose"
  echo "    -O[n] Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
  echo "    -u Compile all ut"
  echo $dotted_line
  echo "    examples, Build ophost ut with O3 level compilation optimization and do not execute."
  echo "    ./build.sh -u --ophost --noexec -O3"
  echo $dotted_line
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    --cov When building uTest locally, count the coverage."
  echo "    --noexec Only compile ut, do not execute the compiled executable file"
  echo "    --make_clean make clean"
  echo "    --asan enable asan on the host side"
  echo "    --ccache=<VALUE> Enable or disable ccache compilation acceleration"
  echo "                     VALUE options: on/off/true/false/disable, Default: on"
  echo "                     Example: --ccache=off to disable ccache"
  echo "    --valgrind run ut with valgrind. This option will disable asan, noexec and run utest by valgrind"
  echo "    --ops Compile specified operator, use snake name, like: --ops=add,add_lora, use ',' to separate different operator"
  echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend910b"
  echo "    --soc supported parameters must only in [ascend910b ascend910_93 ascend950 ascend310p ascend910 ascend310b ascend630 ascend610lite ascend031 ascend035 kirinx90 kirin9030 mc62], A3(--soc=ascend910_93)"
  echo "    --vendor_name Specify the custom operator package vendor name, like: --vendor_name=customize, default to custom"
  echo "    --aicpu build aicpu task"
  echo "    --noaicpu build noaicpu task"
  echo "    --opgraph build opgraph_math.so"
  echo "    --onnxplugin build oponnx_plugin_math.so"
  echo "    --tfplugin build liboptf_plugin_math.so"
  echo "    --opapi build opapi_math.so"
  echo "    --ophost build ophost_math.so"
  echo "    --opkernel build binary kernel"
  echo "    --opkernel_aicpu build aicpu kernel"
  echo "    --pkg build run package with kernel bin"
  echo "    --pkg-type=<TYPE> Specify package type(TYPE options: run/rpm/deb), Default: run"
  echo "    --build-type specify build-type (Type options: Release/Debug), Default:Release"
  echo "    --static build static library package"
  echo "    --experimental Build experimental version"
  echo "    --run_example Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
  echo "    --simulator     Enable simulator mode for run_example (requires --soc parameter)"
  echo "    --genop Create the initial directory for op"
  echo "    --genop_aicpu Create the initial directory for AI CPU op"
  echo "    --mssanitizer Build with mssanitizer mode on the kernel side, with options: '-g --cce-enable-sanitizer'"
  echo "    --oom Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
  echo "    --dump_cce Dump kernel precompiled files"
  echo "    --bisheng_flags Specify bisheng compiler config, like: --bisheng_flags=ccec_g,oom, use ',' to separate different compiler flags"
  echo "    --kernel_template_input Specify kernel template input arguments, like: --kernel_template_input='args0=args0;args1=args1', use ';' to separate different kernel template args"
  echo "    --gtest_filter=pattern Specify gtest filter pattern for ut execution, like: --gtest_filter=AddTest*"
  echo "to be continued ..."
}

check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_test_command=false
  local has_build_command=false
  local has_package=false
  local has_opkernel=false
  local has_opkernel_aicpu=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -u) has_u=true ;;
      --ophost | --opapi | --opgraph | --onnxplugin | --tfplugin)
        has_test_command=true
        has_build_command=true
        ;;
      --pkg) has_package=true ;;
      --opkernel) has_opkernel=true ;;
      --opkernel_aicpu) has_opkernel_aicpu=true ;;
      --help | -h) ;;
    esac
  done

  if [[ "$has_package" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --pkg cannot be used with test(-u), --ophost, --opapi, or --opgraph"
    return 1
  fi

  if [[ "$has_opkernel" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --opkernel cannot be used with test(-u), --ophost, --opapi, or --opgraph"
    return 1
  fi

  if [[ "$has_opkernel_aicpu" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --opkernel_aicpu cannot be used with test(-u), --ophost, --opapi, or --opgraph"
    return 1
  fi

  return 0
}

check_param() {
  if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
    ENABLE_CUSTOM=FALSE
  fi
  if [[ -n "$COMPILED_OPS" && "$ENABLE_TEST" == "FALSE" ]] && [[ "$OP_HOST" == "TRUE" || "$OP_API" == "TRUE" || "$OP_GRAPH" == "TRUE" ]]; then
    echo "[ERROR] --ops cannot be used with --ophost, --opapi, or --opgraph"
    exit 1
  fi

  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with -u"
      exit 1
    fi
    if [[ "$OP_HOST" == "TRUE" || "$OP_API" == "TRUE" || "$OP_GRAPH" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with --ophost, --opapi, --opgraph"
      exit 1
    fi
    if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with --genop"
      exit 1
    fi
    if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
      echo "[ERROR] --pkg cannot be used with --genop_aicpu"
      exit 1
    fi
  fi

  if [[ -n "${BUILD_TYPE}" ]]; then
    if [[ "${BUILD_TYPE}" != "Release" && "${BUILD_TYPE}" != "Debug" ]]; then
      echo "[ERROR] --build-type only support Release/Debug Mode"
      exit 1
    fi
  fi

  if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$ENABLE_OOM" == "TRUE" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
      echo "[ERROR] --build-type=Debug cannot be used with --mssanitizer, --oom, --dump_cce"
      exit 1
    fi
  fi

  if [ -n "$BISHENG_FLAGS" ]; then
    if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$ENABLE_OOM" == "TRUE" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
      echo "[ERROR] --bisheng_flags= cannot be used with --mssanitizer, --oom, --dump_cce"
      exit 1
    fi
  fi

  if [ -n "$KERNEL_TEMPLATE_INPUT" ]; then
    if [[ -z "${COMPILED_OPS}" || "$COMPILED_OPS" == *","* ]]; then
      echo "[ERROR] --kernel_template_input must be used with --ops= and can only specify a single operator"
      exit 1
    fi
  fi

  if [[ "$ENABLE_MSSANITIZER" == "TRUE" && "$ENABLE_OOM" == "TRUE" ]]; then
    echo "[ERROR] --mssanitizer cannot be used with --oom"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "static") && [[ "$ENABLE_PACKAGE" != "TRUE" ]]; then
    echo "[ERROR] --static can only be used with --pkg"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel") && $(echo ${USE_CMD} | grep -wq "jit"); then
    echo "[ERROR] --opkernel cannot be used with --jit"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel_aicpu") && $(echo ${USE_CMD} | grep -wq "jit"); then
    echo "[ERROR] --opkernel_aicpu cannot be used with --jit"
    exit 1
  fi

  if [[ "$ENABLE_SIMULATOR" == "TRUE" && -z "$COMPUTE_UNIT" ]]; then
    echo "[ERROR] --simulator requires --soc parameter to be specified"
    exit 1
  fi

  if [[ "$PACKAGE_TYPE_SET" == "TRUE" && "$ENABLE_PACKAGE" != "TRUE" ]]; then
    echo "[ERROR] --pkg-type can only be used with --pkg"
    exit 1
  fi

  if [[ "$PACKAGE_TYPE" != "run" && "$PACKAGE_TYPE" != "all" ]]; then
    if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
      echo "[ERROR] --pkg-type=${PACKAGE_TYPE} cannot be used with --static"
      exit 1
    fi
    if [[ "$ENABLE_JIT" == "TRUE" ]]; then
      echo "[ERROR] --pkg-type=${PACKAGE_TYPE} cannot be used with --jit"
      exit 1
    fi
    if [[ "$ENABLE_CUSTOM" == "TRUE" ]]; then
      echo "[ERROR] --pkg-type=${PACKAGE_TYPE} only supports built-in ops-math packages; do not use --ops, --vendor_name, or --experimental"
      exit 1
    fi
  fi

  if [[ "$ENABLE_SIMULATOR" == "TRUE" && "$EXAMPLE_MODE" == "graph" ]]; then
    echo "[ERROR] --simulator does not support graph mode. Please use eager mode instead."
    exit 1
  fi
}

set_create_libs() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    BUILD_LIBS=("ophost_${REPOSITORY_NAME}" "opapi_${REPOSITORY_NAME}" "opgraph_${REPOSITORY_NAME}" "oponnx_plugin_${REPOSITORY_NAME}" "optf_plugin_${REPOSITORY_NAME}")
    ENABLE_CREATE_LIB=TRUE
  else
    if [[ "$OP_HOST" == "TRUE" ]]; then
      BUILD_LIBS+=("ophost_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_API" == "TRUE" ]]; then
      BUILD_LIBS+=("opapi_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_GRAPH" == "TRUE" ]]; then
      BUILD_LIBS+=("opgraph_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$ONNX_PLUGIN" == "TRUE" ]]; then
      BUILD_LIBS+=("oponnx_plugin_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$TF_PLUGIN" == "TRUE" ]]; then
      BUILD_LIBS+=("optf_plugin_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_KERNEL" == "TRUE" ]]; then
      ENABLE_BINARY=TRUE
    fi
  fi
}

set_ut_mode() {
  if [[ "$ENABLE_TEST" != "TRUE" ]]; then
    return
  fi
  UT_TEST_ALL=TRUE
  if [[ "$OP_HOST" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_API" == "TRUE" ]]; then
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_GRAPH" == "TRUE" ]]; then
    OP_GRAPH_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL" == "TRUE" ]]; then
    OP_KERNEL_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL_AICPU" == "TRUE" ]]; then
    OP_KERNEL_AICPU_UT=TRUE
    UT_TEST_ALL=FALSE
  fi

  if [[ "$UT_TEST_ALL" == "FALSE" && "$OP_HOST_UT" == "FALSE" && "$OP_API_UT" == "FALSE" && "$OP_GRAPH_UT" == "FALSE" && "$OP_KERNEL_UT" == "FALSE" && "$OP_KERNEL_AICPU_UT" == "FALSE" ]]; then
    echo "[ERROR] At least one test target must be specified (use -u with one of: --ophost, --opapi, --opgraph, --opkernel, --opkernel_aicpu)"
    usage
    exit 1
  fi

  UT_TARGETS=()
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_AICPU_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_aicpu_op_kernel_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
    UT_TARGETS+=("${REPOSITORY_NAME}_op_graph_ut")
  fi
}

process_genop() {
  local opt_name=$1
  local genop_value=$2

  if [[ "$opt_name" == "genop" ]]; then
    ENABLE_GENOP=TRUE
  elif [[ "$opt_name" == "genop_aicpu" ]]; then
    ENABLE_GENOP_AICPU=TRUE
  else
    usage "genop"
    exit 1
  fi

  if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/" ]]; then
    usage "$opt_name"
    exit 1
  fi

  GENOP_NAME=${genop_value##*/}
  local remaining=${genop_value%/*}

  if [[ "$remaining" != *"/"* ]]; then
    GENOP_TYPE=$remaining
    GENOP_BASE=${BASE_PATH}
  else
    GENOP_TYPE=${remaining##*/}
    GENOP_BASE=${remaining%/*}
    if [[ ! "$GENOP_BASE" =~ ^/ && ! "$GENOP_BASE" =~ ^[a-zA-Z]: ]]; then
      GENOP_BASE="${BASE_PATH}/${GENOP_BASE}"
    fi
  fi
}

checkopts_run_example() {
  ENABLE_RUN_EXAMPLE=TRUE
  EXAMPLE_NAME="${!OPTIND}"
  ((OPTIND++))
  if [[ $OPTIND -le $# ]] && [[ "${!OPTIND}" != --* ]]; then
    EXAMPLE_MODE="${!OPTIND}"
    ((OPTIND++))
  fi

  if [[ $OPTIND -le $# ]] && [[ "${!OPTIND}" != --* ]]; then
    PKG_MODE="${!OPTIND}"
    ((OPTIND++))
    if [[ $OPTIND -le $# ]] && [[ "${!OPTIND}" == --vendor_name* ]]; then
      VENDOR="${!OPTIND}"
      VENDOR="${VENDOR#*=}"
      ((OPTIND++))
    else
      VENDOR="custom"
    fi
  fi
}

checkopts() {
  THREAD_NUM=8
  VERBOSE=""
  BUILD_MODE=""
  COMPILED_OPS=""
  UT_TEST_ALL=FALSE
  CHANGED_FILES=""
  CI_MODE=FALSE
  COMPUTE_UNIT=""
  VENDOR_NAME=""
  SHOW_HELP=""
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  SINGLE_EXAMPLE=""
  BUILD_TYPE="Release"
  PACKAGE_TYPE="run"
  PACKAGE_TYPE_SET=FALSE
  USE_CMD="$*"
  BISHENG_FLAGS=""
  KERNEL_TEMPLATE_INPUT=""
  MODULE_EXT=""
  GTEST_FILTER=""

  ENABLE_MSSANITIZER=FALSE
  ENABLE_OOM=FALSE
  ENABLE_DUMP_CCE=FALSE
  ENABLE_COVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=FALSE
  ENABLE_VALGRIND=FALSE
  ENABLE_BINARY=FALSE
  ENABLE_CUSTOM=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_TEST=FALSE
  ENABLE_EXPERIMENTAL=FALSE
  ENABLE_STATIC=FALSE
  ENABLE_JIT=FALSE
  ENABLE_SIMULATOR=FALSE
  ENABLE_RULE_LAUNCH=""
  AICPU_ONLY=FALSE
  DISABLE_AICPU=FALSE
  OP_API_UT=FALSE
  OP_HOST_UT=FALSE
  OP_GRAPH_UT=FALSE
  OP_KERNEL_UT=FALSE
  OP_KERNEL_AICPU_UT=FALSE
  OP_API=FALSE
  OP_HOST=FALSE
  OP_GRAPH=FALSE
  ONNX_PLUGIN=FALSE
  TF_PLUGIN=FALSE
  OP_KERNEL=FALSE
  OP_KERNEL_AICPU=FALSE
  ENABLE_CREATE_LIB=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  NO_FORCE=FALSE
  ENABLE_CCACHE=TRUE
  BUILD_LIBS=()
  UT_TARGETS=()

  ENABLE_GENOP=FALSE
  ENABLE_GENOP_AICPU=FALSE
  GENOP_TYPE=""
  GENOP_NAME=""
  GENOP_BASE=${BASE_PATH}

  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then
      if ! check_option_validity "$arg"; then
        echo "[ERROR] Invalid param $arg, Use 'bash build.sh --help' for more information."
        exit 1
      fi
      if [[ "$arg" == "--pkg-type" ]]; then
        echo "[ERROR] --pkg-type requires a value: run/rpm/deb/all"
        exit 1
      fi
      if [[ "$arg" == --pkg-type=* ]]; then
        check_pkg_type "${arg#*=}"
      fi
    fi
  done

  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      check_help_combinations "$@"
      local comb_result=$?
      if [ $comb_result -eq 1 ]; then
        exit 1
      fi
      SHOW_HELP="general"
      for prev_arg in "$@"; do
        case "$prev_arg" in
          --pkg) SHOW_HELP="package" ;;
          --opkernel) SHOW_HELP="opkernel" ;;
          --opkernel_aicpu) SHOW_HELP="opkernel_aicpu" ;;
          -u) SHOW_HELP="test" ;;
          --make_clean) SHOW_HELP="clean" ;;
          --valgrind) SHOW_HELP="valgrind" ;;
          --ophost) SHOW_HELP="ophost" ;;
          --opapi) SHOW_HELP="opapi" ;;
          --opgraph) SHOW_HELP="opgraph" ;;
          --onnxplugin) SHOW_HELP="onnxplugin" ;;
          --tfplugin) SHOW_HELP="tfplugin" ;;
          --run_example) SHOW_HELP="run_example" ;;
          --genop) SHOW_HELP="genop" ;;
          --genop_aicpu) SHOW_HELP="genop_aicpu" ;;
        esac
      done
      usage "$SHOW_HELP"
      exit 0
    fi
  done

  while getopts $SUPPORTED_SHORT_OPTS opt; do
    case "${opt}" in
      h)
        usage
        exit 0
        ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      O) BUILD_MODE="-O$OPTARG" ;;
      u) ENABLE_TEST=TRUE ;;
      f)
        CHANGED_FILES=$OPTARG
        CI_MODE=TRUE
        ;;
      -) case $OPTARG in
        help)
          usage
          exit 0
          ;;
        ops=*)
          COMPILED_OPS=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        genop=*)
          process_genop "genop" "${OPTARG#*=}"
          ;;
        genop_aicpu=*)
          process_genop "genop_aicpu" "${OPTARG#*=}"
          ;;
        soc=*)
          COMPUTE_UNIT=${OPTARG#*=}
          ;;
        vendor_name=*)
          VENDOR_NAME=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        build-type=*)
          BUILD_TYPE=${OPTARG#*=}
          ;;
        pkg-type=*)
          PACKAGE_TYPE=${OPTARG#*=}
          check_pkg_type "${PACKAGE_TYPE}"
          PACKAGE_TYPE_SET=TRUE
          ;;
        module_extension=*)
          MODULE_EXT=${OPTARG#*=}
          ;;
        mssanitizer) ENABLE_MSSANITIZER=TRUE ;;
        oom) ENABLE_OOM=TRUE ;;
        dump_cce) ENABLE_DUMP_CCE=TRUE ;;
        bisheng_flags=*)
          BISHENG_FLAGS=${OPTARG#*=}
          ;;
        kernel_template_input=*)
          KERNEL_TEMPLATE_INPUT=${OPTARG#*=}
          ;;
        cann_3rd_lib_path=*)
          CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
          ;;
        cov) ENABLE_COVERAGE=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        aicpu) AICPU_ONLY=TRUE ;;
        noaicpu) DISABLE_AICPU=TRUE ;;
        pkg)
          ENABLE_BINARY=TRUE
          ENABLE_PACKAGE=TRUE
          ;;
        static)
          ENABLE_STATIC=TRUE
          ENABLE_BINARY=TRUE
          ;;
        jit) ENABLE_JIT=TRUE ;;
        asan) ENABLE_ASAN=TRUE ;;
        valgrind)
          ENABLE_VALGRIND=TRUE
          ENABLE_UT_EXEC=FALSE
          ;;
        simulator) ENABLE_SIMULATOR=TRUE ;;
        rule_launch=*)
          ENABLE_RULE_LAUNCH=${OPTARG#*=}
          ;;
        ccache=*)
          local ccache_val=${OPTARG#*=}
          if [[ "$ccache_val" == "off" || "$ccache_val" == "false" || "$ccache_val" == "disable" ]]; then
            ENABLE_CCACHE=FALSE
          fi
          ;;
        gtest_filter=*)
          GTEST_FILTER=${OPTARG#*=}
          ;;
        example_name=*) SINGLE_EXAMPLE=${OPTARG#*=} ;;
        run_example)
          checkopts_run_example "$@"
          ;;
        experimental)
          ENABLE_EXPERIMENTAL=TRUE
          ENABLE_CUSTOM=TRUE
          ;;
        make_clean)
          clean_build
          clean_build_out
          clean_third_party
          exit 0
          ;;
        no_force) NO_FORCE=TRUE ;;
        *)
          if ! in_array "$OPTARG" "${RELEASE_TARGETS[@]}"; then
            echo "[ERROR] Invalid option: --$OPTARG"
            usage
            exit 1
          fi

          if [[ "$OPTARG" == "ophost" ]]; then
            OP_HOST=TRUE
          elif [[ "$OPTARG" == "opapi" ]]; then
            OP_API=TRUE
          elif [[ "$OPTARG" == "opgraph" ]]; then
            OP_GRAPH=TRUE
          elif [[ "$OPTARG" == "onnxplugin" ]]; then
            ONNX_PLUGIN=TRUE
          elif [[ "$OPTARG" == "tfplugin" ]]; then
            TF_PLUGIN=TRUE
          elif [[ "$OPTARG" == "opkernel" ]]; then
            OP_KERNEL=TRUE
          elif [[ "$OPTARG" == "opkernel_aicpu" ]]; then
            OP_KERNEL_AICPU=TRUE
          else
            usage
            exit 1
          fi
          ;;
      esac ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
        ;;
    esac
  done
  shift $((OPTIND - 1))
  if [[ "x$@" != "x" ]]; then
    echo "unparsed param: $@"
    usage
    exit
  fi

  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    ENABLE_BINARY=FALSE
  fi

  check_param
  set_create_libs
  set_ut_mode

  if [[ "$CI_MODE" == "TRUE" ]]; then
    run_ci_mode
  fi
}

run_ci_mode() {
  if [[ "$CHANGED_FILES" != /* ]]; then
    CHANGED_FILES=$PWD/$CHANGED_FILES
  fi

  echo "changed files is "$CHANGED_FILES
  echo $dotted_line
  echo "changed lines:"
  cat $CHANGED_FILES
  echo $dotted_line

  local resolve_cmd="python3 scripts/ci/gen_ci_cmd.py -f $CHANGED_FILES --exec --experimental=${ENABLE_EXPERIMENTAL} --pkg=${ENABLE_PACKAGE} --run_example=${ENABLE_RUN_EXAMPLE}"
  if [[ -n "$CANN_3RD_LIB_PATH" && "$CANN_3RD_LIB_PATH" != "${BASE_PATH}/third_party" ]]; then
    resolve_cmd="$resolve_cmd --cann_3rd_lib_path=$CANN_3RD_LIB_PATH"
  fi
  $resolve_cmd
  local ret=$?
  exit $ret
}
