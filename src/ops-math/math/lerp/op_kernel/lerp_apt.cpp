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
 * \file lerp.cpp
 * \brief
 */
#include "arch35/lerp_f16_nddma_without_loops.h"
#include "arch35/lerp_f16_nddma_with_loops.h"
#include "arch35/lerp_f32_nddma_without_loops.h"
#include "arch35/lerp_f32_nddma_with_loops.h"
#include "arch35/lerp_bf16_nddma_without_loops.h"
#include "arch35/lerp_bf16_nddma_with_loops.h"
#include "arch35/lerp_dtype_comb_0_nddma_without_loops.h"
#include "arch35/lerp_dtype_comb_0_nddma_with_loops.h"
#include "arch35/lerp_dtype_comb_1_nddma_without_loops.h"
#include "arch35/lerp_dtype_comb_1_nddma_with_loops.h"

using namespace Lerp;

// start is float16, end is float16, weight is float16, y is float16, max dims in ub is 5 and nddma does not need loops
#define LERP_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY 100000001000100
// start is float16, end is float16, weight is float16, y is float16, max dims in ub is 8 and nddma needs loops
#define LERP_F16_NDDMA_WITH_LOOPS_TILING_KEY 100000001001100
// start is float32, end is float32, weight is float32, y is float32, max dims in ub is 5 and nddma does not need loops
#define LERP_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY 200000001000100
// start is float32, end is float32, weight is float32, y is float32, max dims in ub is 8 and nddma needs loops
#define LERP_F32_NDDMA_WITH_LOOPS_TILING_KEY 200000001001100
// start is bfloat16, end is bfloat16, weight is bfloat16, y is bfloat16, max dims in ub is 5 and nddma does not need
// loops
#define LERP_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY 300000001000100
// start is bfloat16, end is bfloat16, weight is bfloat16, y is bfloat16, max dims in ub is 8 and nddma needs loops
#define LERP_BF16_NDDMA_WITH_LOOPS_TILING_KEY 300000001001100
// start is bfloat16, end is bfloat16, weight is float32, y is bfloat16, max dims in ub is 5 and nddma does not need
// loops
#define LERP_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_TILING_KEY 400000001000100
// start is bfloat16, end is bfloat16, weight is float32, y is bfloat16, max dims in ub is 8 and nddma needs loops
#define LERP_DTYPE_COMB_0_NDDMA_WITH_LOOPS_TILING_KEY 400000001001100
// start is float16, end is float16, weight is float32, y is float16, max dims in ub is 5 and nddma does not need loops
#define LERP_DTYPE_COMB_1_NDDMA_WITHOUT_LOOPS_TILING_KEY 500000001000100
// start is float16, end is float16, weight is float32, y is float16, max dims in ub is 8 and nddma needs loops
#define LERP_DTYPE_COMB_1_NDDMA_WITH_LOOPS_TILING_KEY 500000001001100

extern "C" __global__ __aicore__ void lerp(GM_ADDR start, GM_ADDR end, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe tPipe;
    if (TILING_KEY_IS(LERP_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        LerpF16NddmaWithoutLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_F16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        LerpF16NddmaWithLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        LerpF32NddmaWithoutLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_F32_NDDMA_WITH_LOOPS_TILING_KEY)) {
        LerpF32NddmaWithLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        LerpBf16NddmaWithoutLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_BF16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        LerpBf16NddmaWithLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_DTYPE_COMB_0_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        LerpDtypeComb0NddmaWithoutLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_DTYPE_COMB_0_NDDMA_WITH_LOOPS_TILING_KEY)) {
        LerpDtypeComb0NddmaWithLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_DTYPE_COMB_1_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        LerpDtypeComb1NddmaWithoutLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(LERP_DTYPE_COMB_1_NDDMA_WITH_LOOPS_TILING_KEY)) {
        LerpDtypeComb1NddmaWithLoops op;
        op.Init(start, end, weight, y, workspace, &tilingData, &tPipe);
        op.Process();
        return;
    }
    return;
}
