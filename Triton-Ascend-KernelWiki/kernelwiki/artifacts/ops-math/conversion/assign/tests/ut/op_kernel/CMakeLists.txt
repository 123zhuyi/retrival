# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # assign_apt.cpp uses #include "../tensor_move/arch35/tensor_move.h"
    # Create a staging directory with symlink so the relative path resolves via -I
    # With -I .../kernel_dep_staging/op_kernel, "../tensor_move/arch35/tensor_move.h"
    # resolves to .../kernel_dep_staging/tensor_move/arch35/tensor_move.h
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/op_kernel)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/tensor_move/arch35)
    if(NOT EXISTS "${KERNEL_STAGING_DIR}/tensor_move/arch35/tensor_move.h")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${PROJECT_SOURCE_DIR}/conversion/tensor_move/op_kernel/arch35/tensor_move.h
            ${KERNEL_STAGING_DIR}/tensor_move/arch35/tensor_move.h)
    endif()

    set(assign_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/assign_tiling_arch35.cpp
        )
    AddOpTestCase(assign "ascend950"
        "-I${KERNEL_STAGING_DIR}/op_kernel"
        "${assign_tiling_files}")
endif()
