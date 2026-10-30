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
 * \file roll.cpp
 * \brief Roll kernel entry.
 */

#include "roll.h"

template <typename T>
__aicore__ inline void RunRollKernel(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RollTilingData);
    GET_TILING_DATA_WITH_STRUCT(RollTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    RollKernel::Roll<T> op;
    op.Init(x, y, &tilingData, &pipe);
    op.Process();
}

template <uint32_t schMode>
__global__ __aicore__ void roll(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    (void)schMode;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunRollKernel<DTYPE_X>(x, y, tiling);
}

extern "C" __global__ __aicore__ void roll(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunRollKernel<DTYPE_X>(x, y, tiling);
}
