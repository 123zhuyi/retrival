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
 * \file matrix_set_diag_simt.h
 * \brief matrix_set_diag_simt
 */

#ifndef MATRIX_SET_DIAG_SIMT_H
#define MATRIX_SET_DIAG_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "matrix_set_diag_tilingdata.h"
#include "simt_api/asc_simt.h"

constexpr int32_t THREAD_DIM = 2048;

namespace MSD {
using namespace AscendC;

template <typename T>
class MatrixSetDiagSimt {
public:
    __aicore__ inline MatrixSetDiagSimt(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, const MatrixSetDiagTilingData* tilingData);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> mInputGM_;    // GM x
    GlobalTensor<T> mDiagonalGM_; // GM diagonal
    GlobalTensor<T> mOutputGM_;   // GM y

    uint32_t mBlockIdx_;                 // 核号
    const MatrixSetDiagTilingData* mTD_; // tilingData
    uint64_t mLastDimsSize_;
    uint64_t mWidth_;
    uint64_t mHeight_;
    uint64_t mDiagLen_;
};

template <typename T>
__aicore__ inline void MatrixSetDiagSimt<T>::Init(
    GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, const MatrixSetDiagTilingData* tilingData)
{
    mBlockIdx_ = GetBlockIdx();
    mTD_ = tilingData;
    mLastDimsSize_ = mTD_->xRowNum * mTD_->xColNum;
    mWidth_ = mTD_->xColNum;
    mHeight_ = mTD_->xRowNum;
    mDiagLen_ = mTD_->diagLen;
    mInputGM_.SetGlobalBuffer((__gm__ T*)x);
    mDiagonalGM_.SetGlobalBuffer((__gm__ T*)diagonal);
    mOutputGM_.SetGlobalBuffer((__gm__ T*)y);
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void SimtCompute(
    __gm__ T* inputGM, __gm__ T* diagonalGM, __gm__ volatile T* outputGM, uint32_t outputSize, uint32_t blockIdx,
    uint32_t blockNum, uint32_t width, uint32_t height, uint32_t lastDimSize, uint32_t diagLen, uint32_t magic0,
    uint32_t shift0, uint32_t magic1, uint32_t shift1, uint32_t magic2, uint32_t shift2)
{
    for (uint32_t idx = blockIdx * blockDim.x + threadIdx.x; idx < outputSize;
         idx += blockNum * blockDim.x) {
        uint32_t divW = Simt::UintDiv(idx, magic0, shift0);
        uint32_t h = idx - divW * width;                                  // idx % width;
        uint32_t w = divW - Simt::UintDiv(divW, magic1, shift1) * height; //(idx / width) % height;
        if (h == w) {                                                     // set output from diagonal
            uint32_t c = Simt::UintDiv(idx, magic2, shift2);              // idx / lastDimSize;
            uint32_t diagIdx = c * diagLen + h;
            outputGM[idx] = diagonalGM[diagIdx];
        } else { // set output from input
            outputGM[idx] = inputGM[idx];
        }
    }
}

template <typename T>
__aicore__ inline void MatrixSetDiagSimt<T>::Process()
{
    uint32_t blockNum = GetBlockNum(); // 获取到核数
    if (mBlockIdx_ >= blockNum) {
        return;
    }

    uint32_t outputSize = mTD_->mergeDimSize * mLastDimsSize_;
    if (outputSize == 0) {
        return;
    }

    uint32_t magic0 = 0, shift0 = 0, magic1 = 0, shift1 = 0, magic2 = 0, shift2 = 0;
    GetUintDivMagicAndShift(magic0, shift0, static_cast<uint32_t>(mWidth_));
    GetUintDivMagicAndShift(magic1, shift1, static_cast<uint32_t>(mHeight_));
    GetUintDivMagicAndShift(magic2, shift2, static_cast<uint32_t>(mLastDimsSize_));

    asc_vf_call<SimtCompute<T>>(
        dim3(THREAD_DIM), (__gm__ T*)(mInputGM_.GetPhyAddr()), (__gm__ T*)(mDiagonalGM_.GetPhyAddr()),
        (__gm__ volatile T*)(mOutputGM_.GetPhyAddr()), outputSize, mBlockIdx_, blockNum, mWidth_, mHeight_,
        mLastDimsSize_, mDiagLen_, magic0, shift0, magic1, shift1, magic2, shift2);
}

} // namespace MSD
#endif
