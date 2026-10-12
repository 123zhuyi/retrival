/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bias_tiling_key.h"
#include "bias_tiling_data.h"
#include "bias_kernel.h"

using namespace AscendC;

template <uint32_t schMode>
__global__ __aicore__ void bias(GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(BiasTilingData);
    GET_TILING_DATA_WITH_STRUCT(BiasTilingData, tilingData, tiling);
    if (workspace != nullptr) {
        SetSysWorkspace(workspace);
    }
    if (tilingData.elemNum == 0) {
        return;
    }
    TPipe tPipe;
    constexpr uint32_t SCH_MODE_FLOAT32 = 0;
    constexpr uint32_t SCH_MODE_FLOAT16 = 1;
    constexpr uint32_t SCH_MODE_BFLOAT16 = 2;
    if constexpr (schMode == SCH_MODE_FLOAT32) {
        BiasOp::BiasKernel<float, false> op;
        op.Init(x, bias, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if constexpr (schMode == SCH_MODE_FLOAT16) {
        BiasOp::BiasKernel<half, true> op;
        op.Init(x, bias, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if constexpr (schMode == SCH_MODE_BFLOAT16) {
        BiasOp::BiasKernel<bfloat16_t, true> op;
        op.Init(x, bias, y, workspace, &tilingData, &tPipe);
        op.Process();
    }
}
