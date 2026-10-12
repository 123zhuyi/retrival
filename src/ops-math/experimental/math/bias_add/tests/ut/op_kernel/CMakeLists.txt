# ----------------------------------------------------------------------------
# This file is contributed to the CANN Open Software.
#
# Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
# All Rights Reserved.
#
# Author (account):
# - Yang Zhenze <@gcw_5x5Ew5Ms>
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # Host tiling source the kernel UT depends on (for the generated tiling head).
    set(bias_add_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/bias_add_tiling.cpp
        )
    # AddOpTestCase(opName, socVersion, compileOptions, tilingFiles)
    AddOpTestCase(bias_add "ascend910b" "-DDTYPE_X=float" "${bias_add_tiling_files}")
endif()
