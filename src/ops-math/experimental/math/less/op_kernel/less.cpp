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
 * \file less.cpp
 * \brief
 */

#include "less.h"

template <uint32_t schMode>
__global__ __aicore__ void less(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LessTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessTilingData, tilingData, tiling);

    if (tilingData.isTailBlock == 1) {
        NsLess::Less<DTYPE_X1, DTYPE_X2, DTYPE_Y, true> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum, tilingData.bigprocessDataNumComputes,
            tilingData.smallprocessDataNumComputes, tilingData.tailbigprocessDataNumComputes,
            tilingData.tailsmallprocessDataNumComputes);
        op.Process();
    } else {
        NsLess::Less<DTYPE_X1, DTYPE_X2, DTYPE_Y, false> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum, tilingData.bigprocessDataNumComputes,
            tilingData.smallprocessDataNumComputes, tilingData.tailbigprocessDataNumComputes,
            tilingData.tailsmallprocessDataNumComputes);
        op.Process();
    }
}
