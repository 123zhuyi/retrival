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
 * \file reciprocal.cpp
 * \brief
 */

#include "reciprocal.h"

enum class ReciprocalTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void reciprocal(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReciprocalTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReciprocalTilingData, tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        NsReciprocal::Reciprocal<DTYPE_X, DTYPE_Y, true> op;
        op.Init(
            x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        NsReciprocal::Reciprocal<DTYPE_X, DTYPE_Y, false> op;
        op.Init(
            x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum);
        op.Process();
    }
}
