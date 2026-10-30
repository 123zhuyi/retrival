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
 * \file fill.cpp
 * \brief
 */

#include "fill.h"

using namespace NsFill;

template <uint32_t schMode>
__global__ __aicore__ void fill(GM_ADDR dims, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FillTilingData);
    GET_TILING_DATA_WITH_STRUCT(FillTilingData, tiling_data, tiling);

    if (TILING_KEY_IS(0)) {
        if constexpr (std::is_same_v<DTYPE_VALUE, bool> || std::is_same_v<DTYPE_VALUE, int8_t>) {
            KernelFill<int8_t, false> op;
            op.Init(
                dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_VALUE, int64_t>) {
            KernelFill1_INT64<false> op;
            op.Init(
                dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else {
            KernelFill<DTYPE_VALUE, false> op;
            op.Init(
                dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        if constexpr (std::is_same_v<DTYPE_VALUE, bool> || std::is_same_v<DTYPE_VALUE, int8_t>) {
            KernelFill<int8_t, true> op;
            op.Init(
                dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_VALUE, int64_t>) {
            KernelFill1_INT64<true> op;
            op.Init(
                dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        } else {
            KernelFill<DTYPE_VALUE, true> op;
            op.Init(
                dims, value, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreLoopNum,
                tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum, tiling_data.smallCoreTailDataNum,
                tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum);
            op.Process();
        }
    }
}
