/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_SET_DIAG_V2_SIMT_H
#define MATRIX_SET_DIAG_V2_SIMT_H

#include "simt_api/asc_simt.h"

namespace MSD {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t SIMT_THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t SIMT_THREAD_NUM_LAUNCH_BOUND = 2048;
#endif
constexpr uint32_t SIMT_THREAD_NUM_1_2 = SIMT_THREAD_NUM_LAUNCH_BOUND / 2;
constexpr uint32_t SIMT_THREAD_NUM_1_4 = SIMT_THREAD_NUM_LAUNCH_BOUND / 4;
constexpr uint32_t SIMT_THREAD_NUM_1_8 = SIMT_THREAD_NUM_LAUNCH_BOUND / 8;
constexpr uint32_t SIMT_THREAD_NUM_1_16 = SIMT_THREAD_NUM_LAUNCH_BOUND / 16;

// ==================== dispatch (1 template param: N) ====================

template <template <uint32_t> class FuncT, typename... Args>
__aicore__ inline void SimtDispatch(uint32_t workSize, Args... args)
{
    if (workSize <= SIMT_THREAD_NUM_1_16) {
        asc_vf_call<FuncT<SIMT_THREAD_NUM_1_16>::Run>(dim3(SIMT_THREAD_NUM_1_16), args...);
    } else if (workSize <= SIMT_THREAD_NUM_1_8) {
        asc_vf_call<FuncT<SIMT_THREAD_NUM_1_8>::Run>(dim3(SIMT_THREAD_NUM_1_8), args...);
    } else if (workSize <= SIMT_THREAD_NUM_1_4) {
        asc_vf_call<FuncT<SIMT_THREAD_NUM_1_4>::Run>(dim3(SIMT_THREAD_NUM_1_4), args...);
    } else if (workSize <= SIMT_THREAD_NUM_1_2) {
        asc_vf_call<FuncT<SIMT_THREAD_NUM_1_2>::Run>(dim3(SIMT_THREAD_NUM_1_2), args...);
    } else {
        asc_vf_call<FuncT<SIMT_THREAD_NUM_LAUNCH_BOUND>::Run>(dim3(SIMT_THREAD_NUM_LAUNCH_BOUND), args...);
    }
}

// ==================== cut-tail SIMT kernels ====================

template <typename T, typename U, uint32_t NUM_THREADS>
struct SimtCutTailSetDiagByRowFunc {
    static __simt_vf__ __aicore__ __launch_bounds__(NUM_THREADS) inline void Run(U startRow, U endRow, U startCol,
                                                                                 U endCol, U ubStartRow, U ubStartCol,
                                                                                 int32_t endK, U kLen, uint32_t magic,
                                                                                 uint32_t shift, __gm__ T* diag,
                                                                                 __ubuf__ T* x, U xColNum, U maxDiagLen)
    {
        U numRows = endRow - startRow + 1;
        U maxIterations = kLen * numRows;
        for (uint32_t idx = threadIdx.x; idx < maxIterations; idx += blockDim.x) {
            U rowOffset = Simt::UintDiv(idx, magic, shift);
            U kOffset = idx - rowOffset * kLen;

            U gmXRow = startRow + rowOffset;
            int32_t curK = endK - static_cast<int32_t>(kOffset);
            U gmXCol = gmXRow + curK;
            if (gmXCol < startCol || gmXCol > endCol) {
                continue;
            }
            U curX = (rowOffset + startRow - ubStartRow) * xColNum + gmXCol - ubStartCol;
            U diagElem = (curK >= 0) ? gmXRow : gmXCol;
            U curDiag = kOffset * maxDiagLen + diagElem;
            x[curX] = diag[curDiag];
        }
    }
};

template <typename T, typename U, uint32_t NUM_THREADS>
struct SimtCutTailSetDiagByColFunc {
    static __simt_vf__ __aicore__ __launch_bounds__(NUM_THREADS) inline void Run(U startRow, U endRow, U startCol,
                                                                                 U endCol, U ubStartRow, U ubStartCol,
                                                                                 int32_t endK, U kLen, uint32_t magic,
                                                                                 uint32_t shift, __gm__ T* diag,
                                                                                 __ubuf__ T* x, U xColNum, U maxDiagLen)
    {
        U numCols = endCol - startCol + 1;
        U maxIterations = kLen * numCols;
        for (uint32_t idx = threadIdx.x; idx < maxIterations; idx += blockDim.x) {
            U colOffset = Simt::UintDiv(idx, magic, shift);
            U kOffset = idx - colOffset * kLen;

            U gmXCol = startCol + colOffset;
            int32_t curK = endK - static_cast<int32_t>(kOffset);
            U gmXRow = gmXCol - curK;
            if (gmXRow < startRow || gmXRow > endRow) {
                continue;
            }
            U curX = (gmXRow - ubStartRow) * xColNum + startCol - ubStartCol + colOffset;
            U diagElem = (curK >= 0) ? gmXRow : gmXCol;
            U curDiag = kOffset * maxDiagLen + diagElem;
            x[curX] = diag[curDiag];
        }
    }
};

// ==================== no-cut-tail SIMT kernel ====================

template <typename T, uint32_t NUM_THREADS>
struct SimtNoCutTailSetDiagFunc {
    static __simt_vf__ __aicore__ __launch_bounds__(NUM_THREADS) inline void Run(
        int32_t k1, uint32_t kLen, uint32_t magic0, uint32_t shift0, uint32_t magic1, uint32_t shift1, __ubuf__ T* diag,
        __ubuf__ T* x, uint32_t xRowNum, uint32_t xColNum, uint32_t maxDiagLen, uint32_t maxIterations,
        uint32_t tailAxisDataSize, uint32_t tailDiagSize)
    {
        for (uint32_t idx = threadIdx.x; idx < maxIterations; idx += blockDim.x) {
            uint32_t curBatch = Simt::UintDiv(idx, magic0, shift0);
            uint32_t remainder = idx - curBatch * kLen * xRowNum;
            uint32_t curRow = Simt::UintDiv(remainder, magic1, shift1);
            if (curRow >= xRowNum) {
                continue;
            }
            uint32_t kOffset = remainder - curRow * kLen;
            int32_t curK = k1 - static_cast<int32_t>(kOffset);
            int32_t curCol = static_cast<int32_t>(curRow) + curK;
            if (curCol < 0 || curCol >= static_cast<int32_t>(xColNum)) {
                continue;
            }
            uint32_t curX = curBatch * tailAxisDataSize + curRow * xColNum + curCol;
            uint32_t diagOffset = (curK >= 0) ? curRow : static_cast<uint32_t>(curCol);
            uint32_t curDiag = curBatch * tailDiagSize + kOffset * maxDiagLen + diagOffset;
            x[curX] = diag[curDiag];
        }
    }
};

} // namespace MSD

#endif // MATRIX_SET_DIAG_V2_SIMT_H
