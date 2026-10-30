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
 * \file clip_by_value_v2.cpp
 * \brief
 */

#include "clip_by_value_v2.h"

using namespace NsClipByValueV2;

template <uint32_t schMode>
__global__ __aicore__ void clip_by_value_v2(
    GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueV2TilingData, tiling_data, tiling);

    if (TILING_KEY_IS(1)) {
        TPipe pipe;
        KernelClipByValueV2<DTYPE_X, DTYPE_CLIP_VALUE_MIN, DTYPE_CLIP_VALUE_MAX, DTYPE_Y, true> op;
        op.Init(
            x, clip_value_min, clip_value_max, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        TPipe pipe;
        KernelClipByValueV2<DTYPE_X, DTYPE_CLIP_VALUE_MIN, DTYPE_CLIP_VALUE_MAX, DTYPE_Y, false> op;
        op.Init(
            x, clip_value_min, clip_value_max, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.bigCoreLoopNum, tiling_data.smallCoreLoopNum, tiling_data.ubPartDataNum,
            tiling_data.smallCoreTailDataNum, tiling_data.bigCoreTailDataNum, tiling_data.tailBlockNum, &pipe);
        op.Process();
    }
}
