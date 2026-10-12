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
 * \file cast.cpp
 * \brief
 */

#include "cast.h"

using namespace NsCast;

template <uint32_t schMode>
__global__ __aicore__ void cast(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CastTilingData);
    GET_TILING_DATA_WITH_STRUCT(CastTilingData, tiling_data, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same_v<DTYPE_X, bool>) {
            KernelCast0TBuf<int8_t, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_Y, bool>) {
            KernelCast0TBuf<DTYPE_X, int8_t> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else {
            KernelCast0TBuf<DTYPE_X, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(2)) {
        KernelCast1TBuf4B<DTYPE_X, DTYPE_Y> op;
        op.Init(
            x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        if constexpr (std::is_same_v<DTYPE_Y, bool>) {
            KernelCast2TBuf2B<DTYPE_X, int8_t> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else {
            KernelCast2TBuf2B<DTYPE_X, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(4)) {
        KernelCast3TBuf2B<DTYPE_X, DTYPE_Y> op;
        op.Init(
            x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        if constexpr (std::is_same_v<DTYPE_X, bool>) {
            KernelCast1TBuf2B<int8_t, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_Y, bool>) {
            KernelCast1TBuf2B<DTYPE_X, int8_t> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else {
            KernelCast1TBuf2B<DTYPE_X, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(6)) {
        if constexpr (std::is_same_v<DTYPE_X, bool>) {
            KernelCast1TBuf2B1TBuf4B<int8_t, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_Y, bool>) {
            KernelCast1TBuf2B1TBuf4B<DTYPE_X, int8_t> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        } else {
            KernelCast1TBuf2B1TBuf4B<DTYPE_X, DTYPE_Y> op;
            op.Init(
                x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
                tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
                tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(7)) {
        KernelCast3TBuf2B1TBuf4B<DTYPE_X, DTYPE_Y> op;
        op.Init(
            x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        KernelCastTQueBind op;
        op.Init(
            x, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum, &pipe);
        op.Process();
    }
}
