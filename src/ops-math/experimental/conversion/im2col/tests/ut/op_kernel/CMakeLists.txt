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
    set(im2col_supported_soc_versions ascend910b ascend910_93)
    set(im2col_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/im2col_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/im2col_infershape.cpp
    )
    if(ASCEND_COMPUTE_UNIT IN_LIST im2col_supported_soc_versions)
        AddOpsTestCase(
            OP_NAME im2col
            SOC_VERSION ${ASCEND_COMPUTE_UNIT}
            OTHER_COMPILE_OPTIONS ""
            TILING_SRC_FILES ${im2col_tiling_files}
            UT_SRC_FILES
                ${CMAKE_CURRENT_SOURCE_DIR}/test_im2col.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/im2col_kernel_inst.cpp)
    endif()
endif()
