# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(EXPINT_EXAMPLE)

add_compile_options(-std=c++17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if(NOT DEFINED ASCEND_HOST_ARCH)
    execute_process(COMMAND uname -m
        OUTPUT_VARIABLE _UNAME_M
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ASCEND_HOST_ARCH ${_UNAME_M})
endif()
set(ASCEND_HOST_LINUX "${ASCEND_HOST_ARCH}-linux")
message(STATUS "ASCEND_HOST_ARCH: ${ASCEND_HOST_ARCH}")

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

# ============================================================================
# 图模式 (GE IR) 调用示例
# ============================================================================
add_executable(test_geir_expint test_geir_expint.cpp)

find_library(GRAPH_LIBRARY graph
    PATHS
        ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/lib64
        ${ASCEND_PATH}/compiler/lib64/stub
        ${ASCEND_PATH}/lib64
)
find_library(GE_RUNNER_LIBRARY ge_runner
    PATHS
        ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/lib64
        ${ASCEND_PATH}/compiler/lib64/stub
        ${ASCEND_PATH}/lib64
)
find_library(GRAPH_BASE_LIBRARY graph_base
    PATHS
        ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/lib64
)
find_library(GE_COMPILER_LIBRARY ge_compiler
    PATHS
        ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/lib64
        ${ASCEND_PATH}/compiler/lib64
        ${ASCEND_PATH}/lib64
)

if(NOT GRAPH_LIBRARY OR NOT GE_RUNNER_LIBRARY)
    message(FATAL_ERROR "GE IR libraries not found (libgraph/libge_runner). Please check CANN installation.")
endif()

target_link_libraries(test_geir_expint PRIVATE
    ${GRAPH_LIBRARY}
    ${GE_RUNNER_LIBRARY}
)
if(GRAPH_BASE_LIBRARY)
    target_link_libraries(test_geir_expint PRIVATE ${GRAPH_BASE_LIBRARY})
endif()
if(GE_COMPILER_LIBRARY)
    target_link_libraries(test_geir_expint PRIVATE ${GE_COMPILER_LIBRARY})
endif()

find_path(GE_GRAPH_INCLUDE_DIR
    NAMES graph.h
    PATHS
        ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/include/graph
        ${ASCEND_PATH}/compiler/include/graph
        ${ASCEND_PATH}/include/graph
)
find_path(GE_API_INCLUDE_DIR
    NAMES ge_api.h
    PATHS
        ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/include/ge
        ${ASCEND_PATH}/compiler/include/ge
        ${ASCEND_PATH}/include/ge
)

find_path(CUSTOM_OP_PROTO_INCLUDE_DIR
    NAMES expint_proto.h
    PATHS
        ${ASCEND_PATH}/opp/vendors/custom_math/op_proto/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_graph
)

if(NOT CUSTOM_OP_PROTO_INCLUDE_DIR)
    message(FATAL_ERROR "Custom expint_proto.h not found.")
endif()
message(STATUS "Custom op proto include: ${CUSTOM_OP_PROTO_INCLUDE_DIR}")

find_library(CUSTOM_OP_PROTO_LIBRARY cust_opsproto_rt2.0
    PATHS
        ${ASCEND_PATH}/opp/vendors/custom_math/op_proto/lib/linux/${ASCEND_HOST_ARCH}
)

if(CUSTOM_OP_PROTO_LIBRARY)
    message(STATUS "Custom op proto library: ${CUSTOM_OP_PROTO_LIBRARY}")
    target_link_libraries(test_geir_expint PRIVATE ${CUSTOM_OP_PROTO_LIBRARY})
else()
    message(STATUS "Custom op proto library not found, using local proto header only")
endif()

target_include_directories(test_geir_expint PRIVATE
    ${GE_GRAPH_INCLUDE_DIR}
    ${GE_API_INCLUDE_DIR}
    ${ASCEND_PATH}/opp/built-in/op_proto/inc/
    ${CUSTOM_OP_PROTO_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_graph
    ${ASCEND_PATH}/${ASCEND_HOST_LINUX}/include
    ${ASCEND_PATH}/compiler/include
)
