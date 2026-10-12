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
 * \file softshrink_grad.cpp
 * \brief SoftShrinkGrad 算子 kernel 入口
 */

#include "softshrink_grad.h"

template <uint32_t schMode>
__global__ __aicore__ void softshrink_grad(GM_ADDR input_grad, GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftShrinkGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftShrinkGradTilingData, tilingData, tiling);
    (void)workspace;

    AscendC::TPipe pipe;
    if constexpr (schMode == SOFTSHRINKGRAD_TPL_SCH_MODE_0) {
        NsSoftShrinkGrad::SoftShrinkGrad<half, 1> op;
        op.Init(input_grad, input_x, output_y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == SOFTSHRINKGRAD_TPL_SCH_MODE_1) {
        NsSoftShrinkGrad::SoftShrinkGrad<half, 2> op;
        op.Init(input_grad, input_x, output_y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == SOFTSHRINKGRAD_TPL_SCH_MODE_2) {
        NsSoftShrinkGrad::SoftShrinkGrad<float, 1> op;
        op.Init(input_grad, input_x, output_y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == SOFTSHRINKGRAD_TPL_SCH_MODE_3) {
        NsSoftShrinkGrad::SoftShrinkGrad<float, 2> op;
        op.Init(input_grad, input_x, output_y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == SOFTSHRINKGRAD_TPL_SCH_MODE_4) {
        NsSoftShrinkGrad::SoftShrinkGrad<bfloat16_t, 1> op;
        op.Init(input_grad, input_x, output_y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (schMode == SOFTSHRINKGRAD_TPL_SCH_MODE_5) {
        NsSoftShrinkGrad::SoftShrinkGrad<bfloat16_t, 2> op;
        op.Init(input_grad, input_x, output_y, &tilingData, &pipe);
        op.Process();
    }
}
