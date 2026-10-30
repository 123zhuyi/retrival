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
 * \file tanh_grad_apt.cpp
 * \brief
 */
#include "arch35/tanh_grad_f16_nddma_without_loops.h"
#include "arch35/tanh_grad_f16_nddma_with_loops.h"
#include "arch35/tanh_grad_f32_nddma_without_loops.h"
#include "arch35/tanh_grad_f32_nddma_with_loops.h"
#include "arch35/tanh_grad_bf16_nddma_without_loops.h"
#include "arch35/tanh_grad_bf16_nddma_with_loops.h"
#include "arch35/tanh_grad_dtype_comb_0_nddma_without_loops.h"
#include "arch35/tanh_grad_dtype_comb_0_nddma_with_loops.h"
#include "arch35/tanh_grad_dtype_comb_1_nddma_without_loops.h"
#include "arch35/tanh_grad_dtype_comb_1_nddma_with_loops.h"
#include "arch35/tanh_grad_dtype_comb_2_nddma_without_loops.h"
#include "arch35/tanh_grad_dtype_comb_2_nddma_with_loops.h"
#include "arch35/tanh_grad_dtype_comb_3_nddma_without_loops.h"
#include "arch35/tanh_grad_dtype_comb_3_nddma_with_loops.h"
#include "arch35/tanh_grad_dtype_comb_4_nddma_without_loops.h"
#include "arch35/tanh_grad_dtype_comb_4_nddma_with_loops.h"
#include "arch35/tanh_grad_dtype_comb_5_nddma_without_loops.h"
#include "arch35/tanh_grad_dtype_comb_5_nddma_with_loops.h"

using namespace TanhGrad;

// y is float16, dy is float16, z is float16, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY 100000001000100
// y is float16, dy is float16, z is float16, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_F16_NDDMA_WITH_LOOPS_TILING_KEY 100000001001100
// y is float32, dy is float32, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY 200000001000100
// y is float32, dy is float32, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_F32_NDDMA_WITH_LOOPS_TILING_KEY 200000001001100
// y is bfloat16, dy is bfloat16, z is bfloat16, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY 300000001000100
// y is bfloat16, dy is bfloat16, z is bfloat16, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_BF16_NDDMA_WITH_LOOPS_TILING_KEY 300000001001100
// y is float16, dy is float32, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_TILING_KEY 400000001000100
// y is float16, dy is float32, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_DTYPE_COMB_0_NDDMA_WITH_LOOPS_TILING_KEY 400000001001100
// y is float16, dy is bfloat16, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_DTYPE_COMB_1_NDDMA_WITHOUT_LOOPS_TILING_KEY 500000001000100
// y is float16, dy is bfloat16, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_DTYPE_COMB_1_NDDMA_WITH_LOOPS_TILING_KEY 500000001001100
// y is float32, dy is float16, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_DTYPE_COMB_2_NDDMA_WITHOUT_LOOPS_TILING_KEY 600000001000100
// y is float32, dy is float16, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_DTYPE_COMB_2_NDDMA_WITH_LOOPS_TILING_KEY 600000001001100
// y is float32, dy is bfloat16, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_DTYPE_COMB_3_NDDMA_WITHOUT_LOOPS_TILING_KEY 700000001000100
// y is float32, dy is bfloat16, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_DTYPE_COMB_3_NDDMA_WITH_LOOPS_TILING_KEY 700000001001100
// y is bfloat16, dy is float16, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_DTYPE_COMB_4_NDDMA_WITHOUT_LOOPS_TILING_KEY 800000001000100
// y is bfloat16, dy is float16, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_DTYPE_COMB_4_NDDMA_WITH_LOOPS_TILING_KEY 800000001001100
// y is bfloat16, dy is float32, z is float32, max dims in ub is 5 and nddma does not need loops
#define TANH_GRAD_DTYPE_COMB_5_NDDMA_WITHOUT_LOOPS_TILING_KEY 900000001000100
// y is bfloat16, dy is float32, z is float32, max dims in ub is 8 and nddma needs loops
#define TANH_GRAD_DTYPE_COMB_5_NDDMA_WITH_LOOPS_TILING_KEY 900000001001100

extern "C" __global__ __aicore__ void tanh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe tPipe;
    if (TILING_KEY_IS(TANH_GRAD_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradF16NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_F16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradF16NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradF32NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_F32_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradF32NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradBf16NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_BF16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradBf16NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb0NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_0_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb0NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_1_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb1NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_1_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb1NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_2_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb2NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_2_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb2NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_3_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb3NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_3_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb3NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_4_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb4NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_4_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb4NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_5_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb5NddmaWithoutLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(TANH_GRAD_DTYPE_COMB_5_NDDMA_WITH_LOOPS_TILING_KEY)) {
        TanhGradDtypeComb5NddmaWithLoops op;
        op.Init(y, dy, z, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    }
    return;
}
