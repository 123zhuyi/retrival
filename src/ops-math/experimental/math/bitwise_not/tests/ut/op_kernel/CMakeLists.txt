# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    # 算子自己的 tiling 源码文件
    set(bitwise_not_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/bitwise_not_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/bitwise_not_infershape.cpp
        )
    # param1：算子名称（kernel 命名，须 == 算子目录名 / op_kernel/<op>.cpp）  param2：soc 版本
    # param3：典型 dtype 编译选项（int16 主路径）  param4：tiling 源码文件
    AddOpTestCase(bitwise_not "ascend910b" "-DDTYPE_X=int16_t" "${bitwise_not_tiling_files}")
endif()
