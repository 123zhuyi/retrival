# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/stateless_exponential/arch35)

    # stateless_exponential kernel reuses sim_thread_exponential_simt.h, included as
    # "../sim_thread_exponential/arch35/sim_thread_exponential_simt.h" (build/staging layout).
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/random/sim_thread_exponential/op_kernel
        ${KERNEL_STAGING_DIR}/sim_thread_exponential)
    # sim_thread_exponential_simt.h internally includes "../../random_common/...".
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/random/random_common/op_kernel
        ${KERNEL_STAGING_DIR}/random_common)

    set(stateless_exponential_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/stateless_exponential_tiling_arch35.cpp
        ${PROJECT_SOURCE_DIR}/random/random_common/op_host/arch35/random_tiling_arch35.cpp)
    AddOpTestCase(
        stateless_exponential
        "ascend950"
        "-DTestUtDefaultTilingStruct=RandomUnifiedSimtTilingDataStruct -I${KERNEL_STAGING_DIR}/stateless_exponential -I${KERNEL_STAGING_DIR}/stateless_exponential/arch35"
        "${stateless_exponential_tiling_files}")
endif()
