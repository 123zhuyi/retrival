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
 * \file fill_diagonal_v2.cpp
 * \brief fill diagonal V2 kernel fucntion implementation
 */
#include "fill_diagonal_v2_dense.h"
#include "fill_diagonal_v2_sparse.h"

extern "C" __global__ __aicore__ void fill_diagonal_v2(GM_ADDR x, GM_ADDR fill_value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    #ifdef DTYPE_X
    if (TILING_KEY_IS(0)) {
        FillDiagonalV2Sparse::Kernel<DTYPE_X> op;
        op.Init(x, fill_value, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        FillDiagonalV2Dense::Kernel<DTYPE_X, uint8_t> op;
        op.Init(x, fill_value, &tilingData);
        op.Process();
    }
    #endif
}