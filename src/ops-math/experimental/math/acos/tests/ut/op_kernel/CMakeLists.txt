# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

function(add_acos_kernel_ut_case)
    if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend910b")
        return()
    endif()

    list(FIND ASCEND_OP_NAME acos acos_index)
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND acos_index EQUAL -1)
        return()
    endif()

    set(soc_version Ascend910B1)
    set(case_target acos_${soc_version}_cases_obj)
    set(module_target ${OP_KERNEL_MODULE_NAME}_${soc_version}_cases_obj)

    add_library(${case_target} OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/test_acos.cpp)
    target_compile_options(
        ${case_target}
        PRIVATE
            -g
            -include "${CMAKE_CURRENT_SOURCE_DIR}/acos_tiling.h"
            -DUT_SOC_VERSION="${soc_version}"
    )
    target_include_directories(
        ${case_target}
        PRIVATE
            ${GTEST_INCLUDE}
            ${ASCEND_DIR}/include/base/context_builder
            ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel
            ${PROJECT_SOURCE_DIR}/tests/ut/common
            ${OP_TILING_INCLUDE}
            ${dlog_MATH_INCLUDE_DIR}
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/impl/basic_api/
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/
            ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include/basic_api/
    )
    target_link_libraries(
        ${case_target}
        PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            tikicpulib::${soc_version}
            GTest::gtest
    )

    if(NOT TARGET ${module_target})
        add_library(${module_target} OBJECT $<TARGET_OBJECTS:${case_target}>)
    else()
        target_sources(${module_target} PRIVATE $<TARGET_OBJECTS:${case_target}>)
    endif()
    target_link_libraries(
        ${module_target}
        PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            $<TARGET_OBJECTS:${case_target}>
    )

    list(FIND fastOpTestSocVersions "${soc_version}" soc_index)
    if(soc_index EQUAL -1)
        set(fastOpTestSocVersions ${fastOpTestSocVersions} ${soc_version}
            CACHE STRING "fastOp Test SocVersions" FORCE)
    endif()
endfunction()

if(UT_TEST_ALL OR OP_KERNEL_UT)
    add_acos_kernel_ut_case()
endif()
