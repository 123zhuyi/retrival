/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/cummin.h"

__global__ __aicore__ void cummin(GM_ADDR x, GM_ADDR y, GM_ADDR argmin, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tpipe;
    REGISTER_TILING_DEFAULT(CumminRegbaseTilingData);
    GET_TILING_DATA(tilingData, tiling);

#if (defined(DTYPE_X))
    if (TILING_KEY_IS(RN_FULLY_LOAD)) {
        CumminRegbase::Cummin<DTYPE_X, DTYPE_ARGMIN> op(tpipe, tilingData);
        op.Init(x, y, argmin);
        op.ComputeM();
        op.ComputeReservedM(x, y, argmin);
    } else if (TILING_KEY_IS(N_FULLY_LOAD)) {
        CumminRegbase::Cummin<DTYPE_X, DTYPE_ARGMIN> op(tpipe, tilingData);
        op.Init(x, y, argmin);
        op.ComputeR(op.generalProcessInfo);
        op.ComputeReservedM(x, y, argmin);
    } else if (TILING_KEY_IS(N_NOT_FULLY_LOAD)) {
        CumminRegbase::Cummin<DTYPE_X, DTYPE_ARGMIN> op(tpipe, tilingData);
        op.Init(x, y, argmin);
        op.ComputeN(op.generalProcessInfo);
        op.ComputeReservedM(x, y, argmin);
    } else if (TILING_KEY_IS(COMPUTE_SIMT)) {
        CumminRegbase::Cummin<DTYPE_X, DTYPE_ARGMIN> op(tpipe, tilingData);
        op.InitForSimT(x, y, argmin);
        op.ComputeSimT();
    }
#endif
}
