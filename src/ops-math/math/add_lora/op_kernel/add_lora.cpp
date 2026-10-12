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
 * \file add_lora.cpp
 * \brief
 */

#if __CCE_AICORE__ == 200
#include "add_lora_310.h"
#include "add_lora_sparse_310.h"
#include "bgmv_310.h"
#else
#include "add_lora_single_core.h"
#include "add_lora_normal_core.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void add_lora(
    GM_ADDR y, GM_ADDR x, GM_ADDR weightB, GM_ADDR indices, GM_ADDR weightA, GM_ADDR y_out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workspace);
    if (user1 == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

#if __CCE_AICORE__ == 200
    if (TILING_KEY_IS(200000)) {
        AddLoraKernel310 op;
        op.Init(y, x, weightA, weightB, indices, y_out, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    } else if (TILING_KEY_IS(200001)) {
        AddLoraSparse310 op;
        op.Init(y, x, weightA, weightB, indices, y_out, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    } else if (TILING_KEY_IS(200010)) {
        BgmvKernel310 op;
        op.Init(y, x, weightA, weightB, indices, y_out, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    }
#else
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (TILING_KEY_IS(100000)) {
        AddLoraNormalCoreBatchKernel op;
        op.Init(y, x, weightB, indices, weightA, y_out, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    } else if (TILING_KEY_IS(100001)) {
        AddLoraSingleCoreBatchKernel op;
        op.Init(y, x, weightB, indices, weightA, y_out, user1, tilingData, &tPipe);
        op.Process();
        tPipe.Destroy();
    }
#endif
}