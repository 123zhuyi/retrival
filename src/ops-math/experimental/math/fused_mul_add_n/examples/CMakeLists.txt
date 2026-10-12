# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for fused_mul_add_n examples（独立编译，便于本地手动构建/验证）
#
# 产出两个可执行文件：
#   - test_aclnn_fused_mul_add_n : aclnn 两段式调用（链接自定义包 libcust_opapi.so）
#   - test_geir_fused_mul_add_n  : 图模式 GE IR 构图
#
# 注意：仓库 CI 走 `bash build.sh --run_example fused_mul_add_n eager cust --experimental
#       --vendor_name=custom --soc=ascend910b`（见 README「调用说明」）。本 CMakeLists
#       面向本地手动构建场景，自定义包查找优先级：
#         1) -DCUSTOM_OP_VENDOR_DIR=<.../vendors/custom_math>（显式指定，最稳妥）
#         2) build_out/installed_custom_math/.../vendors/custom_math
#         3) 已安装 vendor 路径 .../opp/vendors/custom_math
cmake_minimum_required(VERSION 3.16)

# 本 CMakeLists 仅用于「独立/手动构建」场景（cmake examples/ 或 build.sh --run_example）。
# 当被仓库/算子构建作为子目录纳入时（顶层 CMakeLists 会 glob 所有含 CMakeLists 的子目录并
# add_subdirectory），跳过本目录——否则会与 tests/st 的同名 target test_aclnn_fused_mul_add_n
# 冲突，且 geir 示例会触发仓库严格编译选项（-Werror）。此 guard 确保仓库构建不进入示例目录，
# 仅在独立构建时生效。
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()

project(fused_mul_add_n_examples CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# CANN 根目录
# ============================================================================
if(DEFINED ENV{ASCEND_HOME_PATH} AND EXISTS "$ENV{ASCEND_HOME_PATH}")
    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_OPP_PATH} AND EXISTS "$ENV{ASCEND_OPP_PATH}/..")
    get_filename_component(CANN_ROOT "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH not set; please source <CANN>/set_env.sh first.")
endif()
message(STATUS "CANN_ROOT: ${CANN_ROOT}")

if(DEFINED ENV{ASCEND_OPP_PATH})
    set(ASCEND_OPP_PATH $ENV{ASCEND_OPP_PATH})
else()
    set(ASCEND_OPP_PATH ${CANN_ROOT}/opp)
endif()

set(CANN_INCLUDE_PATH ${CANN_ROOT}/include)
set(CANN_LIB64 ${CANN_ROOT}/lib64)

# ============================================================================
# 自定义算子包（custom_math）解析 —— 仅 aclnn 示例需要
# ============================================================================
set(CUSTOM_OP_VENDOR_DIR "" CACHE PATH "Path to custom_math vendor dir (.../vendors/custom_math)")

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_fused_mul_add_n.h
    PATHS
        ${CUSTOM_OP_VENDOR_DIR}/op_api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../build_out/installed_custom_math/packages/vendors/custom_math/op_api/include
        ${ASCEND_OPP_PATH}/vendors/custom_math/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/custom_math/op_api/include
    NO_DEFAULT_PATH
)
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CUSTOM_OP_VENDOR_DIR}/op_api/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../build_out/installed_custom_math/packages/vendors/custom_math/op_api/lib
        ${ASCEND_OPP_PATH}/vendors/custom_math/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/custom_math/op_api/lib
    NO_DEFAULT_PATH
)

# ============================================================================
# aclnn 示例（两段式）
# ============================================================================
if(CUSTOM_OP_INCLUDE_DIR AND CUSTOM_OP_LIBRARY)
    message(STATUS "Custom op include: ${CUSTOM_OP_INCLUDE_DIR}")
    message(STATUS "Custom op library: ${CUSTOM_OP_LIBRARY}")
    get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)

    add_executable(test_aclnn_fused_mul_add_n test_aclnn_fused_mul_add_n.cpp)
    target_include_directories(test_aclnn_fused_mul_add_n PRIVATE
        ${CUSTOM_OP_INCLUDE_DIR}
        ${CANN_INCLUDE_PATH}
        ${CANN_INCLUDE_PATH}/aclnn
        ${CANN_INCLUDE_PATH}/aclnnop
    )
    target_link_directories(test_aclnn_fused_mul_add_n PRIVATE ${CUSTOM_OP_LIB_DIR} ${CANN_LIB64})
    target_link_libraries(test_aclnn_fused_mul_add_n PRIVATE cust_opapi ascendcl nnopbase)
    target_link_options(test_aclnn_fused_mul_add_n PRIVATE
        "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}" "-Wl,-rpath,${CANN_LIB64}")
else()
    message(WARNING "custom_math vendor (aclnn_fused_mul_add_n.h + libcust_opapi.so) not found; "
                    "skipping test_aclnn_fused_mul_add_n. "
                    "Pass -DCUSTOM_OP_VENDOR_DIR=<.../vendors/custom_math> to enable it.")
endif()

# ============================================================================
# 图模式（GE IR）示例
# ============================================================================
add_executable(test_geir_fused_mul_add_n test_geir_fused_mul_add_n.cpp)
target_include_directories(test_geir_fused_mul_add_n PRIVATE
    ${CANN_INCLUDE_PATH}/external
    ${CANN_INCLUDE_PATH}/graph
    ${CANN_INCLUDE_PATH}/ge
    ${CANN_INCLUDE_PATH}
    ${ASCEND_OPP_PATH}/built-in/op_proto/inc
)
target_link_directories(test_geir_fused_mul_add_n PRIVATE ${CANN_LIB64})
target_link_libraries(test_geir_fused_mul_add_n PRIVATE graph ge_runner graph_base ge_compiler)
target_link_options(test_geir_fused_mul_add_n PRIVATE "-Wl,-rpath,${CANN_LIB64}")
