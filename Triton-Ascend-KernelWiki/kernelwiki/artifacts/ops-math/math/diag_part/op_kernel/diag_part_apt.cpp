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
 * \file diag_part.cpp
 * \brief kernel for DiagPart
 */

#include "arch35/diag_part_gather.h"
#include "arch35/diag_part_simt_simd.h"

#define TILING_KEY_GATHER 1000
#define TILING_KEY_SIMT 2000

using namespace DiagPart;

__global__ __aicore__ void diag_part(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_KEY_GATHER)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(DiagPartTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagPart::DiagPartGather<int64_t> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            DiagPart::DiagPartGather<int32_t> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            DiagPart::DiagPartGather<int16_t> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SIMT)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(DiagPartTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagPart::DiagPartSimt<int64_t> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            DiagPart::DiagPartSimt<int32_t> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            DiagPart::DiagPartSimt<int16_t> op;
            op.Init(x, y, &tilingData, &pipe);
            op.Process();
        }
    }
}
