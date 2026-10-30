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
 * \file cos.cpp
 * \brief
 */

#include "cos.h"

using namespace NsCos;

template <uint32_t schMode>
__global__ __aicore__ void cos(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CosTilingData);
    GET_TILING_DATA_WITH_STRUCT(CosTilingData, tiling_data, tiling);

#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    using ComputeStrategy = HighPerfStrategy;
#else
    using ComputeStrategy = HighPrecStrategy;
#endif

    KernelCos<DTYPE_X, ComputeStrategy> op;
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(1)) {
        op.Init<true>(
            x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
            tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
            tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum, &pipe);
    } else if (TILING_KEY_IS(0)) {
        op.Init<false>(
            x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
            tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
            tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum, &pipe);
    }
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void cos_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* workspace, uint8_t* tiling)
{
    cos<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
