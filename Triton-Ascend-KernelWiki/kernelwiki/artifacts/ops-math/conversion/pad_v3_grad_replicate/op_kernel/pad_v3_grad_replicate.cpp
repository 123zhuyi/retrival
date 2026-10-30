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
 * \file pad_v3_grad_replicate.cpp
 * \brief
 */

#include "pad_v3_grad_replicate_h.h"
#include "pad_v3_grad_replicate_w.h"
#include "pad_v3_grad_replicate_h_w.h"
#include "pad_v3_grad_replicate_h_w_mini.h"
#include "pad_v3_grad_replicate_small_h_large_w.h"
#include "pad_v3_grad_replicate_small_h_large_w_f16.h"
#include "pad_v3_grad_replicate_large_h_small_w.h"
#include "pad_v3_grad_replicate_large_h_small_w_f16.h"
#include "pad_v3_grad_replicate_h_w_large.h"

extern "C" __global__ __aicore__ void pad_v3_grad_replicate(
    GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1110)) {
        PadV3GradReplicateH<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2110)) {
        PadV3GradReplicateH<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3110)) {
        PadV3GradReplicateH<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1101)) {
        PadV3GradReplicateW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2101)) {
        PadV3GradReplicateW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3101)) {
        PadV3GradReplicateW<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1000)) {
        PadV3GradReplicateHWMini<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2000)) {
        PadV3GradReplicateHWMini<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3000)) {
        PadV3GradReplicateHWMini<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1100)) {
        PadV3GradReplicateSmallHLargeW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2100)) {
        PadV3GradReplicateSmallHLargeWF16<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3100)) {
        PadV3GradReplicateSmallHLargeWF16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1010)) {
        PadV3GradReplicateLargeHSmallW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2010)) {
        PadV3GradReplicateLargeHSmallWF16<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3010)) {
        PadV3GradReplicateLargeHSmallWF16<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(1111)) {
        PadV3GradReplicateHWLarge<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(2111)) {
        PadV3GradReplicateHWLarge<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(3111)) {
        PadV3GradReplicateHWLarge<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(11111)) {
        PadV3GradReplicateHW<float> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(22222)) {
        PadV3GradReplicateHW<half> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
    if (TILING_KEY_IS(33333)) {
        PadV3GradReplicateHW<bfloat16_t> op;
        op.Init(tilingData, x, paddings, y, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
}