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
 * \file exp_segsum_grad.cpp
 * \brief ExpSegsumGrad kernel entry for arch35 / Ascend950.
 *        Reuses the ported A2 compute logic; tiling data comes from a POD struct.
 */

#include "exp_segsum_grad.h"

using namespace ExpSegsumGradArch35;

extern "C" __global__ __aicore__ void exp_segsum_grad(GM_ADDR grad_output, GM_ADDR grad_self, GM_ADDR grad_input,
                                                      GM_ADDR workspace, GM_ADDR tiling)
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

    REGISTER_TILING_DEFAULT(ExpSegsumGradTilingDataArch35);
    GET_TILING_DATA_WITH_STRUCT(ExpSegsumGradTilingDataArch35, tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        ExpSegsumGrad<DTYPE_GRAD_OUTPUT, 0> op;
        op.Init(grad_output, grad_self, grad_input, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ExpSegsumGrad<DTYPE_GRAD_OUTPUT, 1> op;
        op.Init(grad_output, grad_self, grad_input, userWS, &tilingData);
        op.Process();
    } else {
        return;
    }
}
