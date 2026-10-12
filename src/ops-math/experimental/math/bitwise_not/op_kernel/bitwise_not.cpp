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
 * \file bitwise_not.cpp
 * \brief
 */

#include "bitwise_not.h"

// DTYPE_X 由 registry-invoke 框架按 def 中 dtype 列表在编译期为每个 dtype 注入一份二进制。
// BOOL 与 INT8 的 DTYPE_X 同为 int8_t，运行时由 tilingData.isBool 区分语义分支。
template <uint32_t schMode>
__global__ __aicore__ void bitwise_not(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BitwiseNotTilingData);
    GET_TILING_DATA_WITH_STRUCT(BitwiseNotTilingData, tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        NsBitwiseNot::BitwiseNot<DTYPE_X, true> op;
        op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
                tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum, tilingData.tailBlockNum, tilingData.isBool, tilingData.lastCoreId,
                tilingData.lastCoreTailDataNum);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        NsBitwiseNot::BitwiseNot<DTYPE_X, false> op;
        op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
                tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum, tilingData.tailBlockNum, tilingData.isBool, tilingData.lastCoreId,
                tilingData.lastCoreTailDataNum);
        op.Process();
    }
}
