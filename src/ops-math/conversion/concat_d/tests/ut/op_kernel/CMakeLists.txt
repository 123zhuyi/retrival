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
    set(concat_d_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/concat_d_tiling_arch35.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../concat/op_host/arch35/concat_tiling_arch35.cpp
        )
    # concat_d_apt.cpp includes "../concat/arch35/..." relative to its own directory (concat_d/op_kernel/).
    # Create a staging directory in the build tree with a symlink so that path resolves via -I fallback:
    #   STAGING/concat_d/op_kernel/../concat/arch35/ -> STAGING/concat_d/concat/arch35/ -> concat/op_kernel/arch35/
    set(KERNEL_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/kernel_dep_staging)
    file(MAKE_DIRECTORY ${KERNEL_STAGING_DIR}/concat_d/op_kernel)
    set(SYMLINK_TARGET ${CMAKE_CURRENT_SOURCE_DIR}/../../../../concat/op_kernel)
    get_filename_component(SYMLINK_TARGET ${SYMLINK_TARGET} ABSOLUTE)
    if(NOT EXISTS "${KERNEL_STAGING_DIR}/concat_d/concat")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${SYMLINK_TARGET}
            ${KERNEL_STAGING_DIR}/concat_d/concat)
    endif()
    AddOpTestCase(concat_d "ascend950" "-DDTYPE_X=float -DORIG_DTYPE_X=DT_FLOAT -I${KERNEL_STAGING_DIR}/concat_d/op_kernel" "${concat_d_tiling_files}")
endif()
