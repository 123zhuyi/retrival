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
 * \file roll_apt.cpp
 * \brief
 */

#include "arch35/roll_simd.h"
#include "arch35/roll_simt.h"
#include "arch35/roll_unaligned_simd.h"
#include "arch35/roll_hsplit.h"
#include "arch35/roll_gather_simd.h"

#define TILING_KEY_FOR_SIMD_ONE_DIM 10000
#define TILING_KEY_FOR_SIMD_BEFOR_H 20000
#define TILING_KEY_FOR_SIMD_AFTER_H_ALIGN 30000
#define TILING_KEY_FOR_SIMD_AFTER_H_UNALIGN 30001 // 非对齐,可以走传统SIMD分支或切分到H轴分支
#define TILING_KEY_FOR_SIMD_SPLIT_W 40000
#define TILING_KEY_FOR_SIMD_SMALL_TAIL_SHIFTW 50000
#define TILING_KEY_FOR_SIMD_SMALL_TAIL_NOT_SHIFTW 50001
#define TILING_KEY_FOR_SIMD_NONE_TENSOR 60000 // 空tensor

using namespace Roll;

__global__ __aicore__ void roll(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(RollTilingData);
    GET_TILING_DATA_WITH_STRUCT(RollTilingData, tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_FOR_SIMD_SMALL_TAIL_SHIFTW)) {
        RollGatherSimd<DTYPE_X, true> rollSimd(&pipe, &tilingData);
        rollSimd.Init(x, y, workspace);
        rollSimd.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FOR_SIMD_SMALL_TAIL_NOT_SHIFTW)) {
        RollGatherSimd<DTYPE_X, false> rollSimd(&pipe, &tilingData);
        rollSimd.Init(x, y, workspace);
        rollSimd.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FOR_SIMD_BEFOR_H)) {
        RollUnaliegnedSimd<DTYPE_X> rollSimd(&pipe, &tilingData);
        rollSimd.Init(x, y, workspace);
        rollSimd.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FOR_SIMD_AFTER_H_ALIGN)) {
        RollHSplitSimd<DTYPE_X> rollSimd(&pipe, &tilingData);
        rollSimd.Init(x, y, workspace);
        rollSimd.Process();
    } else if (
        TILING_KEY_IS(TILING_KEY_FOR_SIMD_AFTER_H_UNALIGN) || TILING_KEY_IS(TILING_KEY_FOR_SIMD_SPLIT_W) ||
        TILING_KEY_IS(TILING_KEY_FOR_SIMD_ONE_DIM)) {
        RollSimd<DTYPE_X> rollSimd(&pipe, &tilingData);
        rollSimd.Init(x, y, workspace);
        rollSimd.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FOR_SIMD_NONE_TENSOR)) {
    }
}
