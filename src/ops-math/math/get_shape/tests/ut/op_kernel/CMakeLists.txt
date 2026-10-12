# ----------------------------------------------------------------------------
# This program is free software; you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    file(GLOB GET_SHAPE_UT_CASES ${CMAKE_CURRENT_SOURCE_DIR}/test_get_shape*.cpp)
    AddOpsTestCase(
        OP_NAME get_shape
        SOC_VERSION "ascend950"
        OTHER_COMPILE_OPTIONS ""
        TILING_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arch35/get_shape_tiling_arch35.cpp
        UT_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/arch35/get_shape.cpp ${GET_SHAPE_UT_CASES}
    )
endif()
