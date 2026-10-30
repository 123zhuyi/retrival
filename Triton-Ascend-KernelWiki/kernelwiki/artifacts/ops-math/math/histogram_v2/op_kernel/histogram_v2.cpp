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
 * \file histogram_v2.cpp
 * \brief
 */
#include "histogram_v2_scalar.h"

extern "C" __global__ __aicore__ void histogram_v2(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;
    if (TILING_KEY_IS(0)) {
        HistogramV2NS::HistogramV2Scalar<float, float, float> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        HistogramV2NS::HistogramV2Scalar<int32_t, int32_t, int32_t> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        HistogramV2NS::HistogramV2Scalar<int8_t, int8_t, int32_t> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        HistogramV2NS::HistogramV2Scalar<uint8_t, uint8_t, int32_t> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        HistogramV2NS::HistogramV2Scalar<int16_t, int16_t, int32_t> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        HistogramV2NS::HistogramV2Scalar<int32_t, int64_t, int64_t> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        HistogramV2NS::HistogramV2Scalar<half, half, float> op;
        op.Init(x, min, max, y, workspace, &tilingData, &tpipe);
        op.Process();
    }
}