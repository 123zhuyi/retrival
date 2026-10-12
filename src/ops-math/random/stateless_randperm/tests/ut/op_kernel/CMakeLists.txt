# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    # Kernel code uses #include "../../sort/arch35/sort_tiling_data.h" etc., which is designed
    # for the flat staging directory produced by kernel_src_copy during production builds.
    # In UT the kernel is compiled in-place (with the extra op_kernel/ level), so we create a
    # mirrored staging layout in the build directory and pass it via -I so the compiler can
    # resolve the relative include paths on fallback.
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/stateless_randperm/arch35)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/math/sort/op_kernel
        ${KERNEL_STAGING_DIR}/sort)

    set(stateless_randperm_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/stateless_randperm_tiling_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/stateless_randperm_tiling_for_sort.cpp)
    AddOpTestCase(
        stateless_randperm
        "ascend950"
        "-DDTYPE_Y=int32_t -DASCENDC_TPL_KERNEL -I${KERNEL_STAGING_DIR}/stateless_randperm/arch35"
        "${stateless_randperm_tiling_files}")
endif()
