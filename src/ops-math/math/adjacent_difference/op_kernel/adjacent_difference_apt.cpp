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
 * \file adjacent_difference.cpp
 * \brief
 */

#include "./arch35/adjacent_difference_kernel.h"

extern "C" __global__ __aicore__ void adjacent_difference(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(AdjacentDifferenceTilingData, tilingDataIn, tiling);
    const AdjacentDifferenceTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(1)) {
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            AdjacentDifferenceKernel<int64_t, DTYPE_X, DTYPE_Y> op(&pipe);
            op.Init(x, y, workspace, tilingData);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            AdjacentDifferenceKernel<int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
            op.Init(x, y, workspace, tilingData);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            AdjacentDifferenceKernel<int16_t, DTYPE_X, DTYPE_Y> op(&pipe);
            op.Init(x, y, workspace, tilingData);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            AdjacentDifferenceKernel<int8_t, DTYPE_X, DTYPE_Y> op(&pipe);
            op.Init(x, y, workspace, tilingData);
            op.Process();
        } 
    }
}