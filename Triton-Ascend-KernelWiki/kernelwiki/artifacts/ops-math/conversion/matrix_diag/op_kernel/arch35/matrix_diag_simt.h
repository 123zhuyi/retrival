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
 * \file matrix_diag_simt.h
 * \brief
 */
#ifndef MATRIX_DIAG_SIMT_H
#define MATRIX_DIAG_SIMT_H

#include <type_traits>
#include "kernel_operator.h"
#include "matrix_diag_tiling_data_struct.h"
#include "simt_api/asc_simt.h"

namespace MatrixDiagAsc {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
constexpr uint32_t HALF_THREAD_NUM_LAUNCH_BOUND = 256;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
constexpr uint32_t HALF_THREAD_NUM_LAUNCH_BOUND = 1024;
#endif
constexpr uint16_t DIMS_1 = 1;
constexpr uint16_t DIMS_2 = 2;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t NUM_TWO = 2;
template <typename T>
class MatrixDiagSIMT {
    public:
        __aicore__ inline MatrixDiagSIMT(){};
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MatrixDiagSimtTilingData* tilingData, TPipe* pipe);
        __aicore__ inline void Process();
    private:
        uint32_t blockIdx_ = 0;
        GlobalTensor<T> xGm_;
        GlobalTensor<T> yGm_;
        TPipe* pipe_;
        TQue<QuePosition::VECOUT, 1> outQue_;
        uint32_t batchSize_;
        uint32_t nSize_;
        uint32_t halfUbSize_;
        uint32_t realCoreNum_;
        uint32_t mainCoreNum_;
        uint32_t curCoreElements_;
        uint32_t perCoreElements_;//算偏移用
        uint32_t threadNum_;
        uint32_t curCoreLastOneLoopElements_;
        uint32_t curCoreLoopNum_;
        uint32_t maxOneLoopElement_;
        __aicore__ inline void ParseSIMTTilingData(const MatrixDiagSimtTilingData* tilingData);
        __aicore__ inline void ResetUnifiedBufferZero(uint32_t xBlockOffset, uint32_t curLoopElement, uint32_t loopIdx);
};
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(HALF_THREAD_NUM_LAUNCH_BOUND) void MatrixDiagSIMTCompute(
    uint32_t nSize, uint32_t xBaseIdx, uint32_t curCoreElements, uint32_t magic0, uint32_t shift0, __gm__ T* x, __gm__ volatile T* y)
{
    for (uint32_t idx = threadIdx.x; idx < curCoreElements; idx += blockDim.x) {
        uint32_t xIdx = xBaseIdx + idx;
        uint32_t xDivVal = Simt::UintDiv(xIdx, magic0, shift0);
        uint64_t yIdx = (xIdx - xDivVal * nSize) * (nSize + 1) + xDivVal * (nSize * nSize);
        y[yIdx] = x[xIdx];
    }
}

template <typename T>
__aicore__ inline void MatrixDiagSIMT<T>::Init(GM_ADDR x, GM_ADDR y, const MatrixDiagSimtTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    this->ParseSIMTTilingData(tilingData);
    threadNum_ = HALF_THREAD_NUM_LAUNCH_BOUND;
    //计算核内ub切分（刷0用）
    maxOneLoopElement_ = halfUbSize_ / sizeof(T);
    curCoreLoopNum_ = curCoreElements_ * nSize_ / maxOneLoopElement_;
    curCoreLastOneLoopElements_ = curCoreElements_ * nSize_ - curCoreLoopNum_ * maxOneLoopElement_;
    pipe_->InitBuffer(outQue_, DOUBLE_BUFFER, this->halfUbSize_);
}

template <typename T>
__aicore__ inline void MatrixDiagSIMT<T>::ParseSIMTTilingData(const MatrixDiagSimtTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    nSize_ = tilingData->nSize;
    halfUbSize_ = tilingData->ubSize / 2;
    realCoreNum_ = tilingData->realCoreNum;
    mainCoreNum_ = tilingData->mainBlockCount;
    perCoreElements_ = tilingData->mainBlockFactor;
    if (blockIdx_ >= mainCoreNum_) {
        curCoreElements_ = tilingData->tailBlockFactor;
    } else {
        curCoreElements_ = tilingData->mainBlockFactor;
    }
}

template <typename T>
__aicore__ inline void MatrixDiagSIMT<T>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    uint32_t xBlockOffset = 0;
    if (blockIdx_ >= mainCoreNum_) {
        xBlockOffset = mainCoreNum_ * perCoreElements_ + (blockIdx_ - mainCoreNum_) * curCoreElements_;
    } else {
        xBlockOffset = blockIdx_ * perCoreElements_;
    }
    for (uint32_t i = 0; i < curCoreLoopNum_; i++) {
        ResetUnifiedBufferZero(xBlockOffset, maxOneLoopElement_, i);
    }
    if (curCoreLastOneLoopElements_ > 0) {
        ResetUnifiedBufferZero(xBlockOffset, curCoreLastOneLoopElements_, curCoreLoopNum_);
    }
    //同步操作
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
    //快速除
    uint32_t magic0 = 0;
    uint32_t shift0 = 0;
    GetUintDivMagicAndShift(magic0, shift0, nSize_);
    //SMIT搬移
    asc_vf_call<MatrixDiagSIMTCompute<T>>(
        dim3(threadNum_), nSize_, xBlockOffset, curCoreElements_, magic0, shift0, (__gm__ T*) (xGm_.GetPhyAddr()), (__gm__ volatile T*) (yGm_.GetPhyAddr()));
}

template <typename T>
__aicore__ inline void MatrixDiagSIMT<T>::ResetUnifiedBufferZero(uint32_t xBlockOffset, uint32_t curLoopElement, uint32_t loopIdx)
{
    uint64_t yOffset = xBlockOffset * nSize_ + maxOneLoopElement_ * loopIdx;
    //UB对应位置set全0
    T zeroNumber = 0;
    LocalTensor<T> zeroTensor = outQue_.AllocTensor<T>();
    Duplicate(zeroTensor, zeroNumber, curLoopElement);
    outQue_.EnQue<T>(zeroTensor);
    zeroTensor = outQue_.DeQue<T>();
    //DataCopy搬到对应的GM上
    uint32_t burstLen = curLoopElement * sizeof(T);
    DataCopyExtParams copyOutParam{1, burstLen, 0, 0, 0};
    DataCopyPad(yGm_[yOffset], zeroTensor, copyOutParam);
    outQue_.FreeTensor(zeroTensor);
}
} // namespace MatrixDiag
#endif // MATRIX_DIAG_SIMT_H