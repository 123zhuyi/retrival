/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bernoulli_mask.h"
#include "bernoulli_mask_tiling_key.h"

extern "C" __global__ __aicore__ void bernoulli_mask(GM_ADDR mask, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    // ProcessAliased uses the whole-vector-core SyncAll primitive. Ascend C
    // requires this mixed AIV task type even though no AIC task is launched.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(optiling::BernoulliMaskTilingData);
    GET_TILING_DATA_WITH_STRUCT(optiling::BernoulliMaskTilingData, tilingData, tiling);
    AscendC::TPipe pipe;

    // The AscendC precompiler requires numeric literals in TILING_KEY_IS. Keep
    // these values synchronized with bernoulli_mask_tiling_key.h.
    if (TILING_KEY_IS(1)) {
        BernoulliMask::KernelBernoulliMask<half> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        BernoulliMask::KernelBernoulliMask<float> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        BernoulliMask::KernelBernoulliMask<uint64_t, true> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        BernoulliMask::KernelBernoulliMask<uint8_t> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        BernoulliMask::KernelBernoulliMask<int8_t> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        BernoulliMask::KernelBernoulliMask<int16_t> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        BernoulliMask::KernelBernoulliMask<int32_t> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        BernoulliMask::KernelBernoulliMask<int64_t> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        BernoulliMask::KernelBernoulliMask<bfloat16_t> op;
        op.Init(mask, out, &tilingData, &pipe);
        op.Process();
    }
}
