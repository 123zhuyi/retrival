# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
include(cmake/fetch_cann_cmake.cmake)
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${RULE_LAUNCH})
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${RULE_LAUNCH})

set(PKG_NAME math)
project(${PKG_NAME} VERSION 1.0.0)

init_cann_project()

if(NOT ASCEND_INSTALL_PATH AND DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
endif()

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-math with cann pkg" ON)
else()
  message(STATUS "compile project with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-math with cann source" OFF)
endif()

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()
set(OPS_MATH_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 外部传参
option(ENABLE_ASAN "Enable asan" OFF)
option(ENABLE_VALGRIND "Enable valgrind" OFF)
option(ENABLE_TEST "Enable test" OFF)
option(ENABLE_UT_EXEC "Enable exec ut" OFF)
option(ENABLE_BINARY "Enable build binary" OFF)
option(ENABLE_CUSTOM "Enable build custom" OFF)
option(ENABLE_PACKAGE "Enable build package" OFF)
option(ENABLE_EXPERIMENTAL "Enable experimental module" OFF)
option(ENABLE_COVERAGE "Enable coverage" OFF)
option(ENABLE_MSSANITIZER "Enable mssanitizer mode" OFF)
option(ENABLE_OOM "Enable oom mode" OFF)
option(DISABLE_AICPU "Disable aicpu" OFF)
option(ENABLE_DUMP_CCE "Enable dump_cce mode" OFF)

option(OP_HOST_UT "Enable ophost ut" OFF)
option(OP_API_UT "Enable opapi ut" OFF)
option(OP_GRAPH_UT "Enable graph ut" OFF)
option(OP_KERNEL_UT "Enable kernel ut" OFF)
option(OP_KERNEL_AICPU_UT "Enable aicpu kernel ut" OFF)
option(UT_TEST_ALL "Enable all ut" OFF)
option(DOWNLOAD_OPS_TEST_KIT "Download ops-test-kit repository" OFF)
set(GTEST_FILTER "" CACHE STRING "gtest filter pattern for ut execution")
set(BISHENG_FLAGS "" CACHE STRING "bisheng compiler flags")
set(KERNEL_TEMPLATE_INPUT "" CACHE STRING "kernel template input")
set(BUILD_MODE "" CACHE STRING "build mode -O0/O1/O2/O3")
set(ASCEND_COMPUTE_UNIT "ascend910b" CACHE STRING "soc that need to be compiled")
if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party CACHE STRING "cann third party lib path")
endif()
# 指定编译的算子
set(ASCEND_OP_NAME "" CACHE STRING "operators that designated for compilation")
set(VENDOR_NAME "custom" CACHE STRING "vendor name")
set(ASCEND_ALL_COMPUTE_UNIT "ascend031;ascend035;ascend630;ascend610lite;ascend310b;ascend310p;ascend910;ascend910b;ascend910_93;ascend950;kirinx90;kirin9030;mc62")

if("${BUILD_MODE}" STREQUAL "-O0" OR "${BUILD_MODE}" STREQUAL "-O1" OR
   "${BUILD_MODE}" STREQUAL "-O2" OR "${BUILD_MODE}" STREQUAL "-O3"
  )
  set(COMPILE_OP_MODE ${BUILD_MODE})
else()
  if(ENABLE_TEST)
    set(COMPILE_OP_MODE "-O0")
  else()
    set(COMPILE_OP_MODE "-O2")
  endif()
endif()
if(ENABLE_TEST)
  set(COMPILE_OP_MODE "${COMPILE_OP_MODE} -g")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "cmake build type")
else()
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "cmake build type")
endif()

# Cmake Compile配置
set(CMAKE_CXX_STANDARD 17 CACHE STRING "c++17 is needed for this project")
include(cmake/third_party/opbase.cmake) # 放在dependencies.cmake前面
if(DOWNLOAD_OPS_TEST_KIT)
  include(cmake/third_party/ops_test_kit.cmake)
endif()
include(cmake/dependencies.cmake)
include(cmake/variables.cmake)
include(cmake/opbuild.cmake)
add_cann_third_party(eigen)
add_cann_third_party(protobuf)
add_cann_third_party(json)

if(TARGET host_protoc)
  get_target_property(Protobuf_PROTOC_EXECUTABLE host_protoc IMPORTED_LOCATION)
endif()

include(cmake/func.cmake)
include(cmake/intf_pub_linux.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/ut.cmake)

include(version.cmake)
if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  check_cann_pkg_build_deps("ops_math")
endif()
add_cann_version_info_targets()

# ===================================================
# TF 插件模块初始化（无条件调用，函数内部检查条件）
# ===================================================
init_tf_plugin_modules()

add_subdirectory(common)
if(ENABLE_EXPERIMENTAL)
  add_subdirectory(experimental)
else()
  foreach(ops_category ${OPS_CATEGORY_LIST})
    add_subdirectory("${ops_category}")
  endforeach()
endif()

set(_orig_ops "${ASCEND_OP_NAME}")
list(LENGTH _orig_ops _orig_len)
set(_include_examples FALSE)
if(_orig_len GREATER 0)
  foreach(_op ${_orig_ops})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_op}/CMakeLists.txt)
      set(_include_examples TRUE)
      break()
    endif()
  endforeach()
endif()
if(_include_examples)
    add_subdirectory(examples)
endif()

if(ENABLE_TEST)
  if(UT_TEST_ALL OR OP_GRAPH_UT)
    include(cmake/symbol.cmake)
  endif()
  add_subdirectory(tests/ut)
endif()

check_compiled_ops()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_TEST)
  include(cmake/gen_ops_info.cmake)
  gen_ops_info_and_python()

  include(cmake/symbol.cmake)
  if(ENABLE_CUSTOM)
    # custom package library setting and install
    gen_cust_symbol()
  else()
    gen_norm_symbol()
  endif()
endif()

if(ENABLE_PACKAGE)
  include(cmake/package.cmake)

  if(ENABLE_CUSTOM)
    pack_custom()
  else()
    pack_built_in()
  endif()

  if(ENABLE_STATIC)
      include(cmake/static.cmake)
  endif()
endif()
