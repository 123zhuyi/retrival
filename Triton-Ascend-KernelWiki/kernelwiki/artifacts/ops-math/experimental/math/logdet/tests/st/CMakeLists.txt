# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for logdet ST test
cmake_minimum_required(VERSION 3.10)
project(aclnn_logdet_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_MOCK "Use mock aclnn for testing without real operator" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK aclnn + CPU golden - full test validation without NPU")
else()
    message(STATUS "Building with REAL aclnn - NPU output vs CPU golden comparison")
endif()

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
    set(CUSTOM_OPAPI_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../op_api")
    set(CUSTOM_OPAPI_LIBRARY "")
    set(LOCAL_CUSTOM_OPAPI_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../build/libcust_opapi.so")

    if(EXISTS "${LOCAL_CUSTOM_OPAPI_LIBRARY}")
        set(CUSTOM_OPAPI_LIBRARY "${LOCAL_CUSTOM_OPAPI_LIBRARY}")
        message(STATUS "Using local custom opapi library: ${CUSTOM_OPAPI_LIBRARY}")
    elseif(DEFINED ENV{ASCEND_CUSTOM_OPP_PATH} AND
       EXISTS "$ENV{ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so")
        set(CUSTOM_OPAPI_LIBRARY "$ENV{ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so")
    elseif(EXISTS "${CANN_ROOT}/opp/vendors/custom_math/op_api/lib/libcust_opapi.so")
        set(CUSTOM_OPAPI_LIBRARY "${CANN_ROOT}/opp/vendors/custom_math/op_api/lib/libcust_opapi.so")
    endif()

    if(CUSTOM_OPAPI_LIBRARY AND NOT EXISTS "${LOCAL_CUSTOM_OPAPI_LIBRARY}")
        message(STATUS "Using custom opapi library: ${CUSTOM_OPAPI_LIBRARY}")
    elseif(NOT CUSTOM_OPAPI_LIBRARY)
        message(STATUS "Custom opapi library not found, fallback to system libopapi.so")
    endif()

    message(STATUS "CANN_ROOT: ${CANN_ROOT}")
endif()

add_executable(test_aclnn_logdet test_aclnn_logdet.cpp)
target_compile_definitions(test_aclnn_logdet PRIVATE
    LOGDET_ST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(NOT USE_MOCK)
    target_include_directories(test_aclnn_logdet PRIVATE
        ${CUSTOM_OPAPI_INCLUDE_DIRS}
        ${ASCENDCL_INCLUDE_DIRS}
        ${CANN_INCLUDE_DIRS}
        ${CANN_INCLUDE_DIRS}/aclnn
        ${CANN_INCLUDE_DIRS}/aclnnop
        ${CANN_ROOT}/aarch64-linux/include
        ${CANN_ROOT}/aarch64-linux/include/base
        ${CANN_ROOT}/aarch64-linux/pkg_inc
        ${CANN_ROOT}/aarch64-linux/pkg_inc/base
        ${CANN_ROOT}/aarch64-linux/pkg_inc/op_common
        ${CANN_ROOT}/aarch64-linux/pkg_inc/exe_graph
        ${CANN_ROOT}/aarch64-linux/pkg_inc/graph
    )

    target_link_libraries(test_aclnn_logdet
        ${ASCENDCL_LIBRARIES}
        $<$<BOOL:${CUSTOM_OPAPI_LIBRARY}>:${CUSTOM_OPAPI_LIBRARY}>
        $<$<NOT:$<BOOL:${CUSTOM_OPAPI_LIBRARY}>>:${CANN_LIBRARIES}>
        ${CANN_ROOT}/lib64/libnnopbase.so
    )
endif()

install(TARGETS test_aclnn_logdet DESTINATION bin)
