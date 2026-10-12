# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# SortWithIndex (experimental, ascend910b) examples build.
#
# Targets:
#   - test_aclnn_sort_with_index : aclnn two-stage invocation (built by default; needs the installed
#                                  custom op package's op_api: aclnn_sort_with_index.h + libcust_opapi.so).
#   - test_geir_sort_with_index  : GE graph-mode construction (built when BUILD_GEIR=ON, default ON;
#                                  links GE graph libs; graph build always works, RunGraph needs a
#                                  graph-execution runtime).
#
# Usage: see run.sh in this directory (one-click compile + run aclnn example on NPU).
cmake_minimum_required(VERSION 3.10)
project(sort_with_index_examples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_GEIR "Build the GE graph-mode (geir) example" ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. source the CANN set_env.sh first.")
endif()
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# CANN include/lib roots (arch-specific tree, e.g. aarch64-linux/include).
file(GLOB CANN_ARCH_INCLUDE_DIRS "${CANN_ROOT}/*-linux/include")
set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include ${CANN_ARCH_INCLUDE_DIRS})
set(CANN_LIB64 ${CANN_ROOT}/lib64)

# ----------------------------------------------------------------------------------------------------------
# Resolve the installed custom op package (vendor: sort_with_index_custom_math). The currently-active
# package root is exported (colon-separated) via ASCEND_CUSTOM_OPP_PATH and is the single source of truth,
# so it is searched FIRST; otherwise a stale system custom package could shadow it.
# ----------------------------------------------------------------------------------------------------------
set(CUSTOM_OPP_VENDOR_ROOTS "")
if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH} AND NOT "$ENV{ASCEND_CUSTOM_OPP_PATH}" STREQUAL "")
    string(REPLACE ":" ";" CUSTOM_OPP_VENDOR_ROOTS "$ENV{ASCEND_CUSTOM_OPP_PATH}")
    list(REMOVE_ITEM CUSTOM_OPP_VENDOR_ROOTS "")
endif()

# Default install root for the custom op package; override via SWI_OPP_INSTALL_PATH
# (same path passed to the .run installer's --install-path).
set(SWI_OPP_INSTALL_PATH "$ENV{HOME}/sort_with_index_opp")
if(DEFINED ENV{SWI_OPP_INSTALL_PATH} AND NOT "$ENV{SWI_OPP_INSTALL_PATH}" STREQUAL "")
    set(SWI_OPP_INSTALL_PATH "$ENV{SWI_OPP_INSTALL_PATH}")
endif()

set(CUSTOM_OP_INCLUDE_CANDIDATES "")
set(CUSTOM_OP_LIB_CANDIDATES "")
foreach(vroot ${CUSTOM_OPP_VENDOR_ROOTS})
    list(APPEND CUSTOM_OP_INCLUDE_CANDIDATES "${vroot}/op_api/include")
    list(APPEND CUSTOM_OP_LIB_CANDIDATES "${vroot}/op_api/lib")
endforeach()
list(APPEND CUSTOM_OP_INCLUDE_CANDIDATES
    ${SWI_OPP_INSTALL_PATH}/vendors/sort_with_index_custom_math/op_api/include
    ${CANN_ROOT}/opp/vendors/sort_with_index_custom_math/op_api/include
    ${CANN_ROOT}/opp/vendors/custom_math/op_api/include)
list(APPEND CUSTOM_OP_LIB_CANDIDATES
    ${SWI_OPP_INSTALL_PATH}/vendors/sort_with_index_custom_math/op_api/lib
    ${CANN_ROOT}/opp/vendors/sort_with_index_custom_math/op_api/lib
    ${CANN_ROOT}/opp/vendors/custom_math/op_api/lib)

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_sort_with_index.h
    PATHS ${CUSTOM_OP_INCLUDE_CANDIDATES}
    NO_DEFAULT_PATH)
find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS ${CUSTOM_OP_LIB_CANDIDATES}
    NO_DEFAULT_PATH)

# ==========================================================================================================
# aclnn example
# ==========================================================================================================
if(CUSTOM_OP_INCLUDE_DIR AND CUSTOM_OP_LIBRARY)
    message(STATUS "Found custom op include: ${CUSTOM_OP_INCLUDE_DIR}")
    message(STATUS "Found custom op library: ${CUSTOM_OP_LIBRARY}")
    add_executable(test_aclnn_sort_with_index test_aclnn_sort_with_index.cpp)
    target_include_directories(test_aclnn_sort_with_index PRIVATE
        ${CANN_INCLUDE_DIRS}
        ${CUSTOM_OP_INCLUDE_DIR})
    target_link_libraries(test_aclnn_sort_with_index
        ${CANN_LIB64}/libascendcl.so
        ${CANN_LIB64}/libnnopbase.so
        ${CANN_LIB64}/libopapi.so
        ${CUSTOM_OP_LIBRARY})
    install(TARGETS test_aclnn_sort_with_index DESTINATION bin)
else()
    message(WARNING "Custom op package (aclnn_sort_with_index.h / libcust_opapi.so) not found. "
                    "Install the operator package and set ASCEND_CUSTOM_OPP_PATH before building the "
                    "aclnn example. See run.sh.")
endif()

# ==========================================================================================================
# geir (GE graph mode) example
# ==========================================================================================================
if(BUILD_GEIR)
    # GE / graph headers live in arch-specific include subfolders (ge/, graph/, external/).
    set(GE_INC_DIRS "")
    foreach(arch_inc ${CANN_ARCH_INCLUDE_DIRS})
        list(APPEND GE_INC_DIRS
            ${arch_inc}/ge
            ${arch_inc}/graph
            ${arch_inc}/external)
    endforeach()
    # op_graph proto headers (op::Data from array_ops.h; SortWithIndex from our own proto header).
    set(OP_GRAPH_INC_DIRS
        ${CANN_ROOT}/opp/built-in/op_graph/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/..)

    add_executable(test_geir_sort_with_index test_geir_sort_with_index.cpp)
    target_include_directories(test_geir_sort_with_index PRIVATE
        ${CANN_INCLUDE_DIRS}
        ${GE_INC_DIRS}
        ${OP_GRAPH_INC_DIRS})

    # Link GE graph / IR-build libs (names per CANN lib64).
    set(GEIR_LIBS "")
    foreach(libname graph graph_base ge_runner ge_compiler register)
        find_library(GEIR_LIB_${libname} ${libname} PATHS ${CANN_LIB64} NO_DEFAULT_PATH)
        if(GEIR_LIB_${libname})
            list(APPEND GEIR_LIBS ${GEIR_LIB_${libname}})
        endif()
    endforeach()
    target_link_libraries(test_geir_sort_with_index ${GEIR_LIBS})
    install(TARGETS test_geir_sort_with_index DESTINATION bin)
endif()
