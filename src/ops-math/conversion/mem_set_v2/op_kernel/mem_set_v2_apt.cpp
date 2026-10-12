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
 * \file mem_set_v2.cpp
 * \brief mem_set_v2 kernel
 */

#include "kernel_operator.h"
#include "arch35/mem_set_v2.h"

#define TILINGKEY_10002 10002
#define TILINGKEY_10004 10004
#define TILINGKEY_10008 10008
#define TILINGKEY_10016 10016
#define TILINGKEY_10032 10032
#define TILINGKEY_10064 10064
#define TILINGKEY_10128 10128
#define TILINGKEY_10256 10256

using namespace AscendC;

extern "C" __global__ __aicore__ void mem_set_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILINGKEY_10002)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List2TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List2TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10004)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List4TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List4TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10008)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List8TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List8TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10016)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List16TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List16TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10032)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List32TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List32TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10064)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List64TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List64TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10128)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List128TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List128TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_10256)) {
        GET_TILING_DATA_WITH_STRUCT(MemSetV2List256TilingData, tilingData, tiling);
        MemSetV2::MemSetV2<MemSetV2List256TilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    }

    return;
}