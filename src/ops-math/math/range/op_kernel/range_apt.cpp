/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_apt.cpp
 * \brief
 */

#define INT32_TILING_KEY 1001
#define INT64_TILING_KEY 1002
#define FP32_TILING_KEY 1003
#define FP16_TILING_KEY 1004
#define BF16_TILING_KEY 1005

#include "arch35/range_float.h"
#include "arch35/range_int.h"

using namespace Range;

extern "C" __global__ __aicore__ void range(
    GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    if (TILING_KEY_IS(INT32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(RangeTilingDataInt, tiling_data_in, tiling);
        const RangeTilingDataInt* __restrict tilingData = &tiling_data_in;
        Range::RangeInt<int32_t> op;
        op.Init(start, end, step, out, userWS, tilingData);
        op.Process(tilingData);
    } else if (TILING_KEY_IS(INT64_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(RangeTilingDataInt, tiling_data_in, tiling);
        const RangeTilingDataInt* __restrict tilingData = &tiling_data_in;
        Range::RangeInt<int64_t> op;
        op.Init(start, end, step, out, userWS, tilingData);
        op.Process(tilingData);
    } else if (TILING_KEY_IS(FP32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(RangeTilingDataFloat, tiling_data_in, tiling);
        const RangeTilingDataFloat* __restrict tilingData = &tiling_data_in;
        Range::RangeFloat<float, float> op;
        op.Init(start, end, step, out, userWS, tilingData);
        op.Process(tilingData);
    } else if (TILING_KEY_IS(FP16_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(RangeTilingDataFloat, tiling_data_in, tiling);
        const RangeTilingDataFloat* __restrict tilingData = &tiling_data_in;
        Range::RangeFloat<float, half> op;
        op.Init(start, end, step, out, userWS, tilingData);
        op.Process(tilingData);
    } else if (TILING_KEY_IS(BF16_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(RangeTilingDataFloat, tiling_data_in, tiling);
        const RangeTilingDataFloat* __restrict tilingData = &tiling_data_in;
        Range::RangeFloat<float, bfloat16_t> op;
        op.Init(start, end, step, out, userWS, tilingData);
        op.Process(tilingData);
    }
}
