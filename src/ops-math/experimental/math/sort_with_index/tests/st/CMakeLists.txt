# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for sort_with_index ST test (ascend910b)
cmake_minimum_required(VERSION 3.10)
project(aclnn_sort_with_index_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_MOCK "Use mock aclnn for testing without real operator" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK aclnn + CPU golden - full test validation without NPU")
else()
    message(STATUS "Building with REAL aclnn - NPU output vs CPU golden comparison")
endif()

# Set CANN paths (use official ASCEND_HOME_PATH)
if(NOT USE_MOCK)
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH environment variable is not set! source set_env.sh in cann install directory first.")
    endif()
    if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH directory does not exist!")
    endif()

    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
    set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)

    message(STATUS "CANN_ROOT: ${CANN_ROOT}")
endif()

add_executable(test_aclnn_sort_with_index test_aclnn_sort_with_index.cpp)

# Link Ascend libraries for Real mode
if(NOT USE_MOCK)
    # The experimental --pkg custom package vendor name is "custom_math". When $ASCEND_HOME_PATH is
    # not writable, the package installs to a user path whose vendor root is exported (colon-separated)
    # via ASCEND_CUSTOM_OPP_PATH. That path is the SINGLE source of truth for the currently-active
    # package, so it is searched FIRST; otherwise a stale system custom_math (missing the op_api
    # symbols) would shadow the freshly installed one. The colon-list is split and the empty tail
    # (trailing ':') dropped, so each entry is a real vendor root (.../vendors/custom_math).
    set(CUSTOM_OPP_VENDOR_ROOTS "")
    if(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH} AND NOT "$ENV{ASCEND_CUSTOM_OPP_PATH}" STREQUAL "")
        string(REPLACE ":" ";" CUSTOM_OPP_VENDOR_ROOTS "$ENV{ASCEND_CUSTOM_OPP_PATH}")
        list(REMOVE_ITEM CUSTOM_OPP_VENDOR_ROOTS "")
    endif()

    set(CUSTOM_OP_INCLUDE_CANDIDATES "")
    set(CUSTOM_OP_LIB_CANDIDATES "")
    foreach(vroot ${CUSTOM_OPP_VENDOR_ROOTS})
        list(APPEND CUSTOM_OP_INCLUDE_CANDIDATES "${vroot}/op_api/include")
        list(APPEND CUSTOM_OP_LIB_CANDIDATES "${vroot}/op_api/lib")
    endforeach()
    list(APPEND CUSTOM_OP_INCLUDE_CANDIDATES
        $ENV{HOME}/sort_with_index_custom_opp/vendors/custom_math/op_api/include
        ${CANN_ROOT}/opp/vendors/custom_math/op_api/include
        ${CANN_ROOT}/opp/vendors/sort_with_index_custom/op_api/include
        ${CANN_ROOT}/opp/vendors/customize/op_api/include
        /usr/local/Ascend/opp/vendors/sort_with_index_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/sort_with_index_custom/op_api/include)
    list(APPEND CUSTOM_OP_LIB_CANDIDATES
        $ENV{HOME}/sort_with_index_custom_opp/vendors/custom_math/op_api/lib
        ${CANN_ROOT}/opp/vendors/custom_math/op_api/lib
        ${CANN_ROOT}/opp/vendors/sort_with_index_custom/op_api/lib
        ${CANN_ROOT}/opp/vendors/customize/op_api/lib
        /usr/local/Ascend/opp/vendors/sort_with_index_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/sort_with_index_custom/op_api/lib)

    # Find custom operator include directory (aclnn_sort_with_index.h, shipped by op_api).
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_sort_with_index.h
        PATHS ${CUSTOM_OP_INCLUDE_CANDIDATES}
        NO_DEFAULT_PATH
    )

    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Custom operator include directory not found (aclnn_sort_with_index.h). Please install the operator (op_api) first, or build with -DUSE_MOCK=ON for Mock mode.")
    endif()

    target_include_directories(test_aclnn_sort_with_index PRIVATE
        ${ASCENDCL_INCLUDE_DIRS}
        ${CANN_INCLUDE_DIRS}
        ${CUSTOM_OP_INCLUDE_DIR}
    )

    # Find custom operator library (libcust_opapi.so), prioritizing the active package root.
    find_library(CUSTOM_OP_LIBRARY cust_opapi
        PATHS ${CUSTOM_OP_LIB_CANDIDATES}
        NO_DEFAULT_PATH
    )

    if(NOT CUSTOM_OP_LIBRARY)
        message(FATAL_ERROR "cust_opapi library (libcust_opapi.so) not found in any custom op package path. Install the operator (op_api) first and source its set_env.bash, or build with -DUSE_MOCK=ON.")
    endif()
    message(STATUS "Found custom operator library: ${CUSTOM_OP_LIBRARY}")

    target_link_libraries(test_aclnn_sort_with_index
        ${ASCENDCL_LIBRARIES}
        ${CANN_LIBRARIES}
        ${CANN_ROOT}/lib64/libnnopbase.so
        ${CUSTOM_OP_LIBRARY}
    )
endif()

# Install target
install(TARGETS test_aclnn_sort_with_index DESTINATION bin)
