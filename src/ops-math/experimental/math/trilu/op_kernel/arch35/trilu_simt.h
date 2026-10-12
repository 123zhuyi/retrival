/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TRILU_SIMT_H__
#define __TRILU_SIMT_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trilu_tiling_data.h"
#include "trilu_tiling_key.h"
#include "simt_api/asc_simt.h"

namespace NsTrilu {

using namespace AscendC;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void OpTriluSimt(
    int64_t totalElements, int64_t diagonal, int32_t upper, int64_t h, int64_t w, __gm__ T* x, __gm__ T* y)
{
    int64_t matrixSize = h * w;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < static_cast<uint64_t>(totalElements); index += static_cast<uint64_t>(blockDim.x * gridDim.x)) {
        int64_t matrixOffset = static_cast<int64_t>(index) % matrixSize;
        int64_t row = matrixOffset / w;
        int64_t col = matrixOffset % w;
        T val = x[index];
        bool keep = upper ? (col - row >= diagonal) : (col - row <= diagonal);
        y[index] = keep ? val : static_cast<T>(0);
    }
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const TriluTilingData* tilingData)
{
    int64_t totalElements = tilingData->totalElements;
    __gm__ T* x_gm = (__gm__ T*)x;
    __gm__ T* y_gm = (__gm__ T*)y;
    asc_vf_call<OpTriluSimt<T>>(
        dim3(512), totalElements, tilingData->diagonal, tilingData->upper, tilingData->h, tilingData->w, x_gm, y_gm);
}

} // namespace NsTrilu

#endif // __TRILU_SIMT_H__
