/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file exp_segsum_grad.cpp
 * \brief
 */

#include "exp_segsum_grad.h"

using namespace ExpSegsumGrad;

extern "C" __global__ __aicore__ void exp_segsum_grad(GM_ADDR grad_output, GM_ADDR grad_self, GM_ADDR grad_input,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    
    if (TILING_KEY_IS(0)) {
        ExpSegsumGradND<DTYPE_GRAD_OUTPUT, 0> op;
        op.Init(grad_output, grad_self, grad_input, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        ExpSegsumGradND<DTYPE_GRAD_OUTPUT, 1> op;
        op.Init(grad_output, grad_self, grad_input, workspace, &tilingData);
        op.Process();
    } else {
        return;
    }
}