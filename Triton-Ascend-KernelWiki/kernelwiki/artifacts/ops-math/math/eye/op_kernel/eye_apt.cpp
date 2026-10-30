/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file eye.cpp
 * \brief eye
 */

#include "arch35/eye.h"
#include "arch35/eye_simd.h"

using namespace Eye;
#define EYE_FLOAT_SIMD 1000
#define EYE_FLOAT16_SIMD 1001
#define EYE_UINT8_SIMD 1002
#define EYE_INT8_SIMD 1003
#define EYE_INT32_SIMD 1004
#define EYE_INT16_SIMD 1005
#define EYE_BOOL_SIMD 1006
#define EYE_INT64_SIMD 1007
#define EYE_BFLOAT16_SIMD 1008
#define EYE_FLOAT_NUM_UINT32 2000
#define EYE_FLOAT16_NUM_UINT32 2001
#define EYE_UINT8_NUM_UINT32 2002
#define EYE_INT8_NUM_UINT32 2003
#define EYE_INT32_NUM_UINT32 2004
#define EYE_INT16_NUM_UINT32 2005
#define EYE_BOOL_NUM_UINT32 2006
#define EYE_INT64_NUM_UINT32 2007
#define EYE_BFLOAT16_NUM_UINT32 2008
#define EYE_FLOAT_NUM_INT64 2100
#define EYE_FLOAT16_NUM_INT64 2101
#define EYE_UINT8_NUM_INT64 2102
#define EYE_INT8_NUM_INT64 2103
#define EYE_INT32_NUM_INT64 2104
#define EYE_INT16_NUM_INT64 2105
#define EYE_BOOL_NUM_INT64 2106
#define EYE_INT64_NUM_INT64 2107
#define EYE_BFLOAT16_NUM_INT64 2108

extern "C" __global__ __aicore__ void eye(GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(EYE_FLOAT_SIMD)) {
        EyeKernelSimd<float, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_FLOAT16_SIMD)) {
        EyeKernelSimd<half, uint16_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_UINT8_SIMD)) {
        EyeKernelSimd<uint8_t, uint16_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT8_SIMD)) {
        EyeKernelSimd<int8_t, uint16_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT32_SIMD)) {
        EyeKernelSimd<int32_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT16_SIMD)) {
        EyeKernelSimd<int16_t, uint16_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_BOOL_SIMD)) {
        EyeKernelSimd<int8_t, uint16_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT64_SIMD)) {
        EyeKernelSimd<int64_t, uint64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_BFLOAT16_SIMD)) {
        EyeKernelSimd<bfloat16_t, uint16_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_FLOAT_NUM_UINT32)) {
        EyeKernel<float, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_FLOAT16_NUM_UINT32)) {
        EyeKernel<half, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_UINT8_NUM_UINT32)) {
        EyeKernel<uint8_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT8_NUM_UINT32)) {
        EyeKernel<int8_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT32_NUM_UINT32)) {
        EyeKernel<int32_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT16_NUM_UINT32)) {
        EyeKernel<int16_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_BOOL_NUM_UINT32)) {
        EyeKernel<int8_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT64_NUM_UINT32)) {
        EyeKernel<int64_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_BFLOAT16_NUM_UINT32)) {
        EyeKernel<bfloat16_t, uint32_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_FLOAT_NUM_INT64)) {
        EyeKernel<float, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_FLOAT16_NUM_INT64)) {
        EyeKernel<half, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_UINT8_NUM_INT64)) {
        EyeKernel<uint8_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT8_NUM_INT64)) {
        EyeKernel<int8_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT32_NUM_INT64)) {
        EyeKernel<int32_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT16_NUM_INT64)) {
        EyeKernel<int16_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_BOOL_NUM_INT64)) {
        EyeKernel<int8_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_INT64_NUM_INT64)) {
        EyeKernel<int64_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(EYE_BFLOAT16_NUM_INT64)) {
        EyeKernel<bfloat16_t, int64_t> op(tilingData, pipe);
        op.Init(y);
        op.Process();
    }
}
