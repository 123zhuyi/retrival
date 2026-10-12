# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# experimental 自包含 ascend910b 标准 AscendC kernel UT（tikicpulib CPU 模拟）。
# 被测 kernel: experimental/conversion/zeros_like/op_kernel/zeros_like.cpp（910b，标准顶层 tiling 驱动）。
if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(zeros_like_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/zeros_like_tiling.cpp
        )
    # 显式列出 UT 源：test_zeros_like.cpp（gtest 用例）+ zeros_like_kernel_inst.cpp（kernel 模板实例化 shim）。
    # shim 单独 #include kernel 源码并把 template<int BYTE_KEY> 实例化为 extern "C" 包装，
    # 隔离 kernel_operator.h 与 gtest/<iostream> 的符号冲突（DT_*/FORMAT_ND 宏、std::dec 歧义）。
    AddOpsTestCase(
        OP_NAME zeros_like
        SOC_VERSION ascend910b
        OTHER_COMPILE_OPTIONS ""
        TILING_SRC_FILES ${zeros_like_tiling_files}
        UT_SRC_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/test_zeros_like.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/zeros_like_kernel_inst.cpp)
endif()
