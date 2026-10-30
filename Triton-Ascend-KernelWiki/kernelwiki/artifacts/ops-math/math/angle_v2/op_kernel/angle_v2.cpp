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
 * \file angle_v2.cpp
 * \brief
 */
#include "angle_v2_complex.h"
#include "angle_v2_u8.h"
#include "angle_v2_int.h"
#include "angle_v2.h"

using namespace AngleV2N;
#define KEY_DTYPE_COMPLEX64 1
#define KEY_DTYPE_FP32 2
#define KEY_DTYPE_FP16 3
#define KEY_DTYPE_BOOL 4
#define KEY_DTYPE_UINT8 5
#define KEY_DTYPE_INT8 6
#define KEY_DTYPE_INT16 7
#define KEY_DTYPE_INT32 8
#define KEY_DTYPE_INT64 9
#define KEY_DTYPE_BF16 10

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(KEY_DTYPE_COMPLEX64)) {
        AngleV2N::AngleV2Complex<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_FP32)) {
        AngleV2N::AngleV2<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_FP16)) {
        AngleV2N::AngleV2<half> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_BOOL)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_UINT8)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT8)) {
        AngleV2N::AngleV2Int<int8_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT16)) {
        AngleV2N::AngleV2Int<int16_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT32)) {
        AngleV2N::AngleV2Int<int32_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_DTYPE_INT64)) {
        AngleV2N::AngleV2Int<int64_t, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } 
#if (__NPU_ARCH__ == 3510) // bf16 is only supported by 950
    else if (TILING_KEY_IS(KEY_DTYPE_BF16)) {
        AngleV2N::AngleV2<bfloat16_t> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
#endif
}
