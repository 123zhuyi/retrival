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
 * \file drop_out_v3_grad.cpp
 * \brief
 */

#define DO_MASK_TILING_KEY 100

#include "arch35/drop_out_v3_grad.h"

using namespace DropOutV3Grad;

extern "C" __global__ __aicore__ void drop_out_v3_grad(GM_ADDR grad_y, GM_ADDR mask, GM_ADDR scale, GM_ADDR grad_x,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(DO_MASK_TILING_KEY)) {
        DropOutV3Grad::DropOutV3GradImpl<DTYPE_GRAD_Y> op;
        op.Init(grad_y, mask, scale, grad_x, workspace, &tilingData, &pipe);
        op.Process();
    }
}
