# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
project(bernoulli_mask_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
else()
    set(ASCEND_HOME "/usr/local/Ascend/cann")
endif()

if(DEFINED ENV{BERNOULLI_CUSTOM_VENDOR_ROOT})
    set(CUSTOM_VENDOR_ROOT "$ENV{BERNOULLI_CUSTOM_VENDOR_ROOT}")
else()
    set(CUSTOM_VENDOR_ROOT "${ASCEND_HOME}/opp/vendors/custom_math")
endif()

foreach(required
        "${ASCEND_HOME}/include/acl/acl.h"
        "${ASCEND_HOME}/lib64/libopapi.so"
        "${CUSTOM_VENDOR_ROOT}/op_api/include/aclnn_bernoulli.h"
        "${CUSTOM_VENDOR_ROOT}/op_api/lib/libcust_opapi.so")
    if(NOT EXISTS "${required}")
        message(FATAL_ERROR "Required CANN/custom package file is missing: ${required}")
    endif()
endforeach()

add_executable(test_aclnn_bernoulli test_aclnn_bernoulli.cpp)
target_compile_options(test_aclnn_bernoulli PRIVATE -O2 -Wall -Wextra -Werror)
target_include_directories(test_aclnn_bernoulli PRIVATE
    "${CUSTOM_VENDOR_ROOT}/op_api/include"
    "${ASCEND_HOME}/include"
    "${ASCEND_HOME}/include/aclnnop"
)
target_link_directories(test_aclnn_bernoulli PRIVATE
    "${CUSTOM_VENDOR_ROOT}/op_api/lib"
    "${ASCEND_HOME}/lib64"
)
# libcust_opapi intentionally reuses L0 operators (notably DSAGenBitMask) from
# the installed ops-math libopapi, so --no-as-needed and this link order matter.
target_link_options(test_aclnn_bernoulli PRIVATE -Wl,--no-as-needed)
target_link_libraries(test_aclnn_bernoulli PRIVATE
    cust_opapi
    opapi
    ascendcl
    nnopbase
    dl
    pthread
)
set_target_properties(test_aclnn_bernoulli PROPERTIES
    BUILD_RPATH "${CUSTOM_VENDOR_ROOT}/op_api/lib;${ASCEND_HOME}/lib64"
)
