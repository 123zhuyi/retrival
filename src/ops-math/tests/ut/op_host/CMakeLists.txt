# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

## set variables
set(OP_HOST_UT_EXE ${PKG_NAME}_op_host_ut)

## add optiling ut cases static lib: libmath_op_tiling_ut_cases.a
add_optiling_ut_modules(${OP_TILING_MODULE_NAME})

## add opinfershape ut cases static lib: libmath_op_infershape_ut_cases.a
add_infershape_ut_modules(${OP_INFERSHAPE_MODULE_NAME})

## add onnx framework ut cases static lib: libmath_op_framework_ut_cases.a
add_framework_ut_modules(${OP_FRAMEWORK_MODULE_NAME})

if(EXISTS ${OPS_MATH_DIR}/common/tests/ut/framework/CMakeLists.txt)
    # 全量模式(ASCEND_OP_NAME 空)接入；--ops 模式仅当命中 common 框架插件算子时接入，
    # 避免无关算子 trigger 时白编一坨 common framework UT
    set(_math_include_common_fw OFF)
    if(NOT ASCEND_OP_NAME)
        set(_math_include_common_fw ON)
    else()
        file(GLOB _fw_plugins ${OPS_MATH_DIR}/common/src/framework/*_onnx_plugin.cpp)
        foreach(_p ${_fw_plugins})
            get_filename_component(_n ${_p} NAME_WE)
            string(REGEX REPLACE "_onnx_plugin$" "" _op "${_n}")
            if(_op IN_LIST ASCEND_OP_NAME)
                set(_math_include_common_fw ON)
            endif()
        endforeach()
    endif()
    if(_math_include_common_fw)
        add_subdirectory(${OPS_MATH_DIR}/common/tests/ut/framework
                         ${CMAKE_CURRENT_BINARY_DIR}/common_op_framework)
    endif()
endif()

## add ophost source so: libophost_math_ut.so
add_library(${OPHOST_NAME}_ut SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
    $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
    $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
)
target_link_libraries(${OPHOST_NAME}_ut PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    -Wl,--whole-archive
        rt2_registry_static
    -Wl,--no-whole-archive
    tiling_api
)

## add ophost ut exe:  math_op_host_ut
add_executable(${OP_HOST_UT_EXE}
    test_op_host_main.cpp
)
target_compile_definitions(${OP_HOST_UT_EXE} PRIVATE)
target_compile_options(${OP_HOST_UT_EXE} PUBLIC -fPIE -fno-access-control)
add_dependencies(${OP_HOST_UT_EXE} ${OPHOST_NAME}_ut)
target_include_directories(${OP_HOST_UT_EXE} PRIVATE
    ${ASCEND_DIR}/pkg_inc
    )
target_link_directories(${OP_HOST_UT_EXE} PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    )
target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    -Wl,--whole-archive
        ${OP_TILING_MODULE_NAME}_cases
        ${OP_INFERSHAPE_MODULE_NAME}_cases
        ${OP_FRAMEWORK_MODULE_NAME}_cases
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
        metadef
        unified_dlog
    -Wl,--as-needed
    register
    opp_registry
    graph
    graph_base
    platform
    dl
    GTest::gtest
)

## exec ophost ut exe
if(ENABLE_UT_EXEC)
  if(NOT "${GTEST_FILTER}" STREQUAL "")
    set(GTEST_FILTER_ARG "--gtest_filter=${GTEST_FILTER}")
  else()
    set(GTEST_FILTER_ARG "")
  endif()
  if(ENABLE_ASAN)
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
      OUTPUT_VARIABLE LIBASAN_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE result
    )
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "compiler not support asan, please disable asan")
    endif()

    execute_process(
      COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
      OUTPUT_VARIABLE LIBSTDC_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE stdc_result
    )
    if(NOT stdc_result EQUAL 0)
      message(FATAL_ERROR "not find libstdc++.so, please confirm.")
    endif()

    add_custom_command(
      TARGET ${OP_HOST_UT_EXE}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
        LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}
        ASAN_OPTIONS=detect_leaks=0 $<TARGET_FILE:${OP_HOST_UT_EXE}> ${GTEST_FILTER_ARG}
      COMMENT "Run ops op_host utest with asan"
      )
  else()
    add_custom_command(
      TARGET ${OP_HOST_UT_EXE}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} $<TARGET_FILE:${OP_HOST_UT_EXE}> ${GTEST_FILTER_ARG}
      COMMENT "Run ops op_host utest"
      )
  endif()
endif()

if(ENABLE_VALGRIND)
  add_custom_command(
    TARGET ${OP_HOST_UT_EXE}
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} && valgrind --undef-value-errors=no
      --leak-check=full $<TARGET_FILE:${OP_HOST_UT_EXE}> ${GTEST_FILTER_ARG}
    COMMENT "Run ops op_host utest by valgrind"
    )
endif()
