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
    # tril_apt.cpp uses #include "../triu/arch35/triangulator_base.h"
    # Create a staging directory with symlinks so the relative paths resolve via -I
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/op_kernel)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/triu/arch35)

    # Create symlinks for all required triu headers
    set(TRIU_HEADERS
        triangulator_base.h
        split_all.h
        split_row_and_col.h
        split_tiny_shape.h
        split_medium_shape.h
    )
    foreach(header ${TRIU_HEADERS})
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${PROJECT_SOURCE_DIR}/conversion/triu/op_kernel/arch35/${header}
            ${KERNEL_STAGING_DIR}/triu/arch35/${header})
    endforeach()

    set(tril_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/tril_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../triu/op_host/arch35/triangulator_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../triu/op_host/arch35/triu_tiling.cpp
        )
    AddOpTestCase(tril "ascend950"
        "-DDTYPE_X=float -I${KERNEL_STAGING_DIR}/op_kernel -DUT_CUSTOM_TILING_DATA_KEYS=Tril_4"
        "${tril_tiling_files}")
endif()
