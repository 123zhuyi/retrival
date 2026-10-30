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
 * \file fused_mul_add_n.cpp
 * \brief
 */

#include "arch35/fused_mul_add_n.h"
#include "arch35/fused_mul_add_n_half.h"

using namespace FusedMulAddNOp;

// input_x1 is float32, input_x2 is float32, input_x3 is float32, output_y is float32
#define FUSED_MUL_ADD_F32_TILING_KEY 100000000000100
// input_x1 is float16, input_x2 is float16, input_x3 is float16, output_y is float16
#define FUSED_MUL_ADD_F16_TILING_KEY 200000000000100
// input_x1 is int32, input_x2 is int32, input_x3 is int32, output_y is int32
#define FUSED_MUL_ADD_S32_TILING_KEY 300000000000100
// input_x1 is int16, input_x2 is int16, input_x3 is int16, output_y is int16
#define FUSED_MUL_ADD_S16_TILING_KEY 400000000000100
// input_x1 is bfloat16, input_x2 is bfloat16, input_x3 is bfloat16, output_y is bfloat16
#define FUSED_MUL_ADD_BF16_TILING_KEY 500000000000100

extern "C" __global__ __aicore__ void fused_mul_add_n(
    GM_ADDR inputX1, GM_ADDR inputX2, GM_ADDR inputX3, GM_ADDR outputY, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe tPipe;
    if (TILING_KEY_IS(FUSED_MUL_ADD_F32_TILING_KEY)) {
        FusedMulAddN<float> op;
        op.Init(inputX1, inputX2, inputX3, outputY, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(FUSED_MUL_ADD_F16_TILING_KEY)) {
        FusedMulAddNHalf<half> op;
        op.Init(inputX1, inputX2, inputX3, outputY, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(FUSED_MUL_ADD_S32_TILING_KEY)) {
        FusedMulAddN<int32_t> op;
        op.Init(inputX1, inputX2, inputX3, outputY, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(FUSED_MUL_ADD_S16_TILING_KEY)) {
        FusedMulAddN<int16_t> op;
        op.Init(inputX1, inputX2, inputX3, outputY, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(FUSED_MUL_ADD_BF16_TILING_KEY)) {
        FusedMulAddNHalf<bfloat16_t> op;
        op.Init(inputX1, inputX2, inputX3, outputY, workspace, &tilingData, &tPipe);
        op.Process();
    }
}
