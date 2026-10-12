# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    set(diag_part_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/diag_part_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/diag_part_infershape.cpp
        )
    # Test float16: DTYPE_X=half, DTYPE_VALUE=1(FLOAT16), ALIGN_SIZE=16
    AddOpTestCase(diag_part_fp16 "ascend910b" "-DDTYPE_X=half -DDTYPE_VALUE=1 -DALIGN_SIZE=16" "${diag_part_tiling_files}")
    # Test float32: DTYPE_X=float, DTYPE_VALUE=0(FLOAT), ALIGN_SIZE=8
    AddOpTestCase(diag_part_fp32 "ascend910b" "-DDTYPE_X=float -DDTYPE_VALUE=0 -DALIGN_SIZE=8" "${diag_part_tiling_files}")
    # Test int32: DTYPE_X=int32_t, DTYPE_VALUE=3(INT32), ALIGN_SIZE=8
    AddOpTestCase(diag_part_int32 "ascend910b" "-DDTYPE_X=int32_t -DDTYPE_VALUE=3 -DALIGN_SIZE=8" "${diag_part_tiling_files}")
endif()
