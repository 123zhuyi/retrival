/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file zeros_like_kernel_inst.cpp
 * \brief op_kernel UT 专用：被测 kernel zeros_like 为 template<int BYTE_KEY>，CPU 模拟(tikicpulib)
 *        需要可调用符号。本文件 #include kernel 源码并把 4 个字节宽度桶实例化为非模板 extern "C"
 *        包装入口（zeros_like_1b/2b/4b/8b），供 test_zeros_like.cpp 经 ICPU_RUN_KF 调用。
 *
 *        关键：本文件**不包含 gtest / <iostream>**，从而隔离 kernel_operator.h（含 matmul 实现，
 *        其内部使用无限定 dec）与 std::dec 的符号歧义——该歧义正是直接在 gtest TU 内 #include
 *        kernel 源码时的编译错误根因。
 */
#include "../../../op_kernel/zeros_like.cpp"

// 4 个字节宽度桶的非模板 extern "C" 包装：与 kernel template<int BYTE_KEY> 的 if constexpr 分支一一对应。
extern "C" __global__ __aicore__ void zeros_like_1b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    zeros_like<ZL_KEY_1B>(x, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void zeros_like_2b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    zeros_like<ZL_KEY_2B>(x, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void zeros_like_4b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    zeros_like<ZL_KEY_4B>(x, y, workspace, tiling);
}
extern "C" __global__ __aicore__ void zeros_like_8b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    zeros_like<ZL_KEY_8B>(x, y, workspace, tiling);
}
