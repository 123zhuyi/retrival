/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matrix_diag_part_simt.h
 * \brief SIMT kernel implementation for matrix_diag_part operator
 */

#ifndef MATRIX_DIAG_PART_SIMT_H_
#define MATRIX_DIAG_PART_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "matrix_diag_part_tiling_data.h"
#include "matrix_diag_part_tiling_key.h"

namespace NsMatrixDiagPart {

using namespace AscendC;

template <typename IDX_T>
static constexpr uint32_t THREADS = (sizeof(IDX_T) == 4) ? 1024 : 512;

static constexpr int64_t INT32_MAX_VAL = 0x7FFFFFFFLL;
static constexpr int64_t UINT32_MAX_VAL = 0xFFFFFFFFLL;

template <typename T, typename IDX_T>
__simt_vf__ __aicore__ __launch_bounds__(THREADS<IDX_T>) inline void OpMatrixDiagPartSimt(
    IDX_T totalOutputElements, IDX_T diagLen, IDX_T inputRowStride, IDX_T matrixSize, IDX_T diagMagic, IDX_T diagShift,
    __gm__ T* input, __gm__ T* output)
{
    IDX_T stride = static_cast<IDX_T>(blockDim.x * gridDim.x);
    for (IDX_T idx = static_cast<IDX_T>(blockIdx.x * blockDim.x + threadIdx.x); idx < totalOutputElements;
         idx += stride) {
        IDX_T batchIdx = Simt::UintDiv<IDX_T>(idx, diagMagic, diagShift);
        IDX_T diagIdx = idx - batchIdx * diagLen;
        IDX_T inputOffset = batchIdx * matrixSize + diagIdx * inputRowStride;
        output[idx] = input[inputOffset];
    }
}

template <typename T, typename IDX_T>
__aicore__ inline void DispatchVf(int64_t totalOutputElements, int64_t diagLen, int64_t inputRowStride,
                                  int64_t matrixSize, __gm__ T* xGm, __gm__ T* yGm)
{
    IDX_T diagMagic = 0;
    IDX_T diagShift = 0;
    GetUintDivMagicAndShift<IDX_T>(diagMagic, diagShift, static_cast<IDX_T>(diagLen));
    asc_vf_call<OpMatrixDiagPartSimt<T, IDX_T>>(dim3(THREADS<IDX_T>), static_cast<IDX_T>(totalOutputElements),
                                                static_cast<IDX_T>(diagLen), static_cast<IDX_T>(inputRowStride),
                                                static_cast<IDX_T>(matrixSize), diagMagic, diagShift, xGm, yGm);
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const MatrixDiagPartTilingData* tilingData)
{
    int64_t totalOutputElements = tilingData->totalOutputElements;
    int64_t diagLen = tilingData->diagLen;
    int64_t inputRowStride = tilingData->inputRowStride;
    int64_t matrixSize = tilingData->matrixSize;

    __gm__ T* xGm = (__gm__ T*)x;
    __gm__ T* yGm = (__gm__ T*)y;

    if (totalOutputElements == 0) {
        return;
    }

    int64_t batchTotal = totalOutputElements / diagLen;
    int64_t totalInputElements = batchTotal * matrixSize;
    if (totalOutputElements <= INT32_MAX_VAL && totalInputElements <= UINT32_MAX_VAL) {
        DispatchVf<T, uint32_t>(totalOutputElements, diagLen, inputRowStride, matrixSize, xGm, yGm);
    } else {
        DispatchVf<T, uint64_t>(totalOutputElements, diagLen, inputRowStride, matrixSize, xGm, yGm);
    }
}

} // namespace NsMatrixDiagPart
#endif // MATRIX_DIAG_PART_SIMT_H_
