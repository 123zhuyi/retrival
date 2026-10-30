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
 * \file transpose_v2.cpp
 * \brief
 */

#include "transpose021.h"
#include "transpose0213.h"

using namespace TransposeV2;

extern "C" __global__ __aicore__ void transpose_v2(
    GM_ADDR x, GM_ADDR perm, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
// perm=[0,2,1]
#if ORIG_DTYPE_X == DT_FLOAT16 || (!(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(20)) {
        Transpose021<half> op(&pipe);
        op.Process(x, y, &tiling_data);
    }
#elif ORIG_DTYPE_X == DT_FLOAT32
    if (TILING_KEY_IS(40)) {
        Transpose021<float, true> op(&pipe);
        op.Process(x, y, &tiling_data);
    }
#endif
// 012 and 0213
#if ORIG_DTYPE_X == DT_FLOAT16 || (!(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)) && ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(120)) {
        Transpose102<half> op(&pipe);
        op.Init102(x, y, &tiling_data);
        op.ProcessTransMode(0);
    } else if (TILING_KEY_IS(121)) {
        Transpose102<half> op(&pipe);
        op.Init102(x, y, &tiling_data);
        op.Process102CopyMode(0);
    } else if (TILING_KEY_IS(221)) {
        Transpose0213<half> op(&pipe);
        op.Init0213(x, y, &tiling_data);
        op.Process0213CopyMode();
    }
#elif ORIG_DTYPE_X == DT_FLOAT32
    if (TILING_KEY_IS(140)) {
        Transpose102<float, true> op(&pipe);
        op.Init102(x, y, &tiling_data);
        op.ProcessTransMode(0);
    } else if (TILING_KEY_IS(141)) {
        Transpose102<float> op(&pipe);
        op.Init102(x, y, &tiling_data);
        op.Process102CopyMode(0);
    } else if (TILING_KEY_IS(241)) {
        Transpose0213<float> op(&pipe);
        op.Init0213(x, y, &tiling_data);
        op.Process0213CopyMode();
    }
#endif
}