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
 * \file pow.cpp
 * \brief pow kernel impl
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/pow_tensor_scalar_float.h"
#include "arch35/pow_tensor_scalar_integer.h"
#include "arch35/pow_bf16_nddma_with_loops.h"
#include "arch35/pow_bf16_nddma_without_loops.h"
#include "arch35/pow_f16_nddma_with_loops.h"
#include "arch35/pow_f16_nddma_without_loops.h"
#include "arch35/pow_f32_nddma_with_loops.h"
#include "arch35/pow_f32_nddma_without_loops.h"
#include "arch35/pow_u8_nddma_with_loops.h"
#include "arch35/pow_u8_nddma_without_loops.h"
#include "arch35/pow_s8_nddma_with_loops.h"
#include "arch35/pow_s8_nddma_without_loops.h"
#include "arch35/pow_s16_nddma_with_loops.h"
#include "arch35/pow_s16_nddma_without_loops.h"
#include "arch35/pow_s32_nddma_with_loops.h"
#include "arch35/pow_s32_nddma_without_loops.h"

#define FP16_TILING_KEY 1001
#define BF16_TILING_KEY 2001
#define FP32_TILING_KEY 3001
#define U8_TILING_KEY 4001
#define S8_TILING_KEY 5001
#define S16_TILING_KEY 6001
#define S32_TILING_KEY 7001
// input_x is float16, input_y is float16, output_z is float16, max dims in ub is 5 and nddma does not need loops
#define POW_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY 100000001000100
// input_x is float16, input_y is float16, output_z is float16, max dims in ub is 8 and nddma needs loops
#define POW_F16_NDDMA_WITH_LOOPS_TILING_KEY 100000001001100
// input_x is bfloat16, input_y is bfloat16, output_z is bfloat16, max dims in ub is 5 and nddma does not need loops
#define POW_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY 200000001000100
// input_x is bfloat16, input_y is bfloat16, output_z is bfloat16, max dims in ub is 8 and nddma needs loops
#define POW_BF16_NDDMA_WITH_LOOPS_TILING_KEY 200000001001100
// input_x is float32, input_y is float32, output_z is float32, max dims in ub is 5 and nddma does not need loops
#define POW_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY 300000001000100
// input_x is float32, input_y is float32, output_z is float32, max dims in ub is 8 and nddma needs loops
#define POW_F32_NDDMA_WITH_LOOPS_TILING_KEY 300000001001100
// input_x is uint8, input_y is uint8, output_z is uint8, max dims in ub is 5 and nddma does not need loops
#define POW_U8_NDDMA_WITHOUT_LOOPS_TILING_KEY 400000001000100
// input_x is uint8, input_y is uint8, output_z is uint8, max dims in ub is 8 and nddma needs loops
#define POW_U8_NDDMA_WITH_LOOPS_TILING_KEY 400000001001100
// input_x is int8, input_y is int8, output_z is int8, max dims in ub is 5 and nddma does not need loops
#define POW_S8_NDDMA_WITHOUT_LOOPS_TILING_KEY 500000001000100
// input_x is int8, input_y is int8, output_z is int8, max dims in ub is 8 and nddma needs loops
#define POW_S8_NDDMA_WITH_LOOPS_TILING_KEY 500000001001100
// input_x is int16, input_y is int16, output_z is int16, max dims in ub is 5 and nddma does not need loops
#define POW_S16_NDDMA_WITHOUT_LOOPS_TILING_KEY 600000001000100
// input_x is int16, input_y is int16, output_z is int16, max dims in ub is 8 and nddma needs loops
#define POW_S16_NDDMA_WITH_LOOPS_TILING_KEY 600000001001100
// input_x is int32, input_y is int32, output_z is int32, max dims in ub is 5 and nddma does not need loops
#define POW_S32_NDDMA_WITHOUT_LOOPS_TILING_KEY 700000001000100
// input_x is int32, input_y is int32, output_z is int32, max dims in ub is 8 and nddma needs loops
#define POW_S32_NDDMA_WITH_LOOPS_TILING_KEY 700000001001100

using namespace Pow;

extern "C" __global__ __aicore__ void pow(GM_ADDR base, GM_ADDR exponent, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe tPipe;
    if (TILING_KEY_IS(FP16_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarFloat<half, half> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(BF16_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarFloat<bfloat16_t, bfloat16_t> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(FP32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarFloat<float, float> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(U8_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarInteger<uint8_t, uint8_t> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(S8_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarInteger<int8_t, int8_t> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(S16_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarInteger<int16_t, int16_t> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(S32_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorScalarTilingData, tilingData, tiling);
        PowTensorScalarInteger<int32_t, int32_t> op;
        op.Init(base, exponent, y, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowF16NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_F16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowF16NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_BF16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowBf16NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_BF16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowBf16NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowF32NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_F32_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowF32NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_U8_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowU8NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_U8_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowU8NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_S8_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowS8NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_S8_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowS8NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_S16_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowS16NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_S16_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowS16NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_S32_NDDMA_WITHOUT_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowS32NddmaWithoutLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(POW_S32_NDDMA_WITH_LOOPS_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(PowTensorTensorTilingData, tilingData, tiling);
        PowS32NddmaWithLoops op;
        op.Init(base, exponent, y, workspace, &tilingData, &tPipe);
        op.Process();
    }
}