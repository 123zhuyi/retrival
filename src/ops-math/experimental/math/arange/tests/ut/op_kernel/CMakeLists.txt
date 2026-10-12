# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的 tiling 依赖源码（op_kernel UT 框架要求把 tiling/infershape 一并编入）
    set(arange_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arange_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/arange_infershape.cpp
        )

    # 使用 AddOpTestCase（参考同目录 erf/floor_mod/lin_space_d op_kernel UT 框架）
    # param1：算子名称（kernel 方式命名）
    # param2：soc 版本
    # param3：自定义编译选项 —— arange kernel 模板类（KernelArange / KernelArange_Cast）在
    #         test_arange_kernel.cpp 内直接实例化（fp32/int8/int16 三类薄包装核），
    #         不依赖 -DDTYPE_OUT 单 dtype 宏，故此处留空。
    # param4：该算子依赖的所有 tiling 源码文件
    AddOpTestCase(arange "ascend910b" "" "${arange_tiling_files}")
endif()
