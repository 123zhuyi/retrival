/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file squared_difference.cpp
 * \brief SquaredDifference 算子 kernel 入口
 *        每个 tilingKey 对应唯一的 KernelSquaredDifference<T> 实例，
 *        if constexpr 确保每个编译单元只存在一种 dtype 的 UB buffer，
 *        避免多 dtype 对象同时分配导致 UB 溢出。
 */

#include "squared_difference.h"

template <uint32_t schMode>
__global__ __aicore__ void squared_difference(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SquaredDifferenceTilingData);
    GET_TILING_DATA_WITH_STRUCT(SquaredDifferenceTilingData, tilingData, tiling);
    // Empty outputs must not initialize UB buffers or touch any GM address.
    if (tilingData.totalLength == 0) {
        return;
    }
    TPipe pipe;

    // if constexpr：每个 schMode 实例化只编译其中一个分支，
    // 其余分支在编译期丢弃，不分配 UB 内存。
    if constexpr (schMode == SD_KEY_FP32_ONEDIM || schMode == SD_KEY_FP32_BRC) {
        KernelSquaredDifference<float, float, false> op(&pipe);
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == SD_KEY_FP16_ONEDIM || schMode == SD_KEY_FP16_BRC) {
        KernelSquaredDifference<half, float, true> op(&pipe);
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == SD_KEY_BF16_ONEDIM || schMode == SD_KEY_BF16_BRC) {
        KernelSquaredDifference<bfloat16_t, float, true> op(&pipe);
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if constexpr (schMode == SD_KEY_INT32_ONEDIM || schMode == SD_KEY_INT32_BRC) {
        KernelSquaredDifference<int32_t, int32_t, false> op(&pipe);
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else { // SD_KEY_INT64_ONEDIM || SD_KEY_INT64_BRC
        KernelSquaredDifference<int64_t, int64_t, false> op(&pipe);
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
}
