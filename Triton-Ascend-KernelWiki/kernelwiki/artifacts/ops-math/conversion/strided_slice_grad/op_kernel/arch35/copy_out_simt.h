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
 * \file copy_out_simt.h
 * \brief
 */

#ifndef COPY_OUT_SIMT_H
#define COPY_OUT_SIMT_H

#include "strided_slice_grad_base.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"

namespace StridedSliceGrad
{
using namespace AscendC;
constexpr int16_t SIMT_BEGIN_OFFSET = 8;
constexpr int16_t SIMT_STRIDES_OFFSET = 16;
constexpr int16_t SIMT_FUSEDOUTPUTINNERSHAPE_OFFSET = 24;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void CopyOut(__local_mem__ T* srcLocal, __gm__ T* dstGm,
                                                                    __local_mem__ uint32_t* simtB32Local, __local_mem__ int64_t* simtB64Local,
                                                                    uint32_t curLoopBlockNum, uint32_t blockIdx, uint32_t normalCoreProcessNum,
                                                                    uint32_t loopNum, uint32_t normalLoopBlockNum, uint32_t inputDimNum)
{
    for (int32_t i = threadIdx.x; i < curLoopBlockNum; i += blockDim.x) {
        int32_t idx = blockIdx * normalCoreProcessNum + loopNum * normalLoopBlockNum + i;
        int32_t srcIndex = i;
        int64_t dstIndex = 0;

#pragma unroll
        for (int32_t k = (MAX_SUPPORT_DIM - inputDimNum); k <= DIM0; k++) { // [8 - inputDimNum, 7]
            uint32_t t2 = __umulhi(static_cast<uint32_t>(idx), simtB32Local[k]);
            t2 = t2 + idx;
            uint32_t newIndex = t2 >> simtB32Local[k + MAX_SUPPORT_DIM];
            uint32_t curDimIndex = idx - simtB64Local[k] * newIndex;
            dstIndex += ((simtB64Local[k + MAX_SUPPORT_DIM] + newIndex * simtB64Local[k + MAX_SUPPORT_DIM * 2]) *
                         simtB64Local[k + MAX_SUPPORT_DIM * 3]);
            idx = curDimIndex;
        }
        dstGm[dstIndex] = srcLocal[srcIndex];
    }
}

template <typename T>
class CopyOutSimt : public StridedSliceGradBase
{
public:
    __aicore__ CopyOutSimt() {};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR output, const StridedSliceGradTilingData& tilingData, TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void InitCopyParams();
    __aicore__ inline void ConstructDivParams();

private:
    __aicore__ inline void ProcessPerLoop();

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, DB_BUFFER> dataQueue_;
    TBuf<TPosition::VECCALC> simtB64Buf_;
    TBuf<TPosition::VECCALC> simtB32Buf_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    DataCopyExtParams normalCopyParams_;
    DataCopyExtParams tailCopyParams_;
    DataCopyExtParams curCopyParams_;
    DataCopyPadExtParams<T> padParams_;

    uint32_t curUbLoopCount_;
    uint32_t normalLoopBlockNum_;
    uint32_t tailLoopBlockNum_;
    uint32_t curLoopBlockNum_;
    uint32_t burstLen_;
    uint32_t blockIdx_;
    uint32_t loopNum_;

    uint32_t shift_[MAX_SUPPORT_DIM];
    uint32_t m_[MAX_SUPPORT_DIM];

    constexpr static uint32_t SIMT_BUF_B64_NUM = 32;
    constexpr static uint32_t SIMT_BUF_B32_NUM = 16;
    constexpr static uint32_t SIMT_BUF_B64_SIZE = SIMT_BUF_B64_NUM * sizeof(int64_t);
    constexpr static uint32_t SIMT_BUF_B32_SIZE = SIMT_BUF_B32_NUM * sizeof(uint32_t);

    cce::dim3 threadsPerBlock_{THREAD_DIM};
};

template <typename T>
__aicore__ inline void CopyOutSimt<T>::Init(GM_ADDR dy, GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                          TPipe& pipeIn)
{
    // tiling_data
    StridedSliceGradBase::ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    burstLen_ = bufferSize_ / sizeof(T);
    // shield normal core and tail core
    curCoreProcessNum_ = (GetBlockIdx() + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    curUbLoopCount_ = (curCoreProcessNum_ + burstLen_ - 1) / burstLen_;

    normalLoopBlockNum_ = (curCoreProcessNum_ >= burstLen_) ? burstLen_ : curCoreProcessNum_;
    tailLoopBlockNum_ = curCoreProcessNum_ - (curUbLoopCount_ - 1) * normalLoopBlockNum_;
    InitCopyParams();

    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_;
    srcGm_.SetGlobalBuffer((__gm__ T*)dy + intraCoreOffset);
    dstGm_.SetGlobalBuffer((__gm__ T*)output);

    pipe_ = pipeIn;
    pipe_.InitBuffer(dataQueue_, DB_BUFFER, bufferSize_);
    pipe_.InitBuffer(simtB64Buf_, SIMT_BUF_B64_SIZE);
    pipe_.InitBuffer(simtB32Buf_, SIMT_BUF_B32_SIZE);
    ConstructDivParams();
}

template <typename T>
__aicore__ inline void CopyOutSimt<T>::ConstructDivParams()
{
    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        GetUintDivMagicAndShift<uint32_t>(m_[i], shift_[i], fusedSliceInnerShape_[i]);
    }
}

template <typename T>
__aicore__ inline void CopyOutSimt<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (loopNum_ = 0; loopNum_ < curUbLoopCount_; loopNum_++) {
        // sheild normal block and tail block
        curLoopBlockNum_ = (loopNum_ == (curUbLoopCount_ - 1)) ? tailLoopBlockNum_ : normalLoopBlockNum_;
        curCopyParams_ = (loopNum_ == (curUbLoopCount_ - 1)) ? tailCopyParams_ : normalCopyParams_;
        ProcessPerLoop();
    }
}

template <typename T>
__aicore__ inline void CopyOutSimt<T>::ProcessPerLoop()
{
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, srcGm_[normalLoopBlockNum_ * loopNum_], curCopyParams_, padParams_);
    dataQueue_.EnQue(xLocal);

    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();

    LocalTensor<int64_t> simtB64Local = simtB64Buf_.Get<int64_t>();
    LocalTensor<uint32_t> simtB32Local = simtB32Buf_.Get<uint32_t>();

    for(int i = 0; i < MAX_SUPPORT_DIM; i++) {
        simtB32Local(i) = m_[i];
        simtB32Local(i + MAX_SUPPORT_DIM) = shift_[i];
    }

    for(int i = 0; i < MAX_SUPPORT_DIM; i++) {
        simtB64Local(i) = fusedSliceInnerShape_[i];
        simtB64Local(i + SIMT_BEGIN_OFFSET) = begin_[i];
        simtB64Local(i + SIMT_STRIDES_OFFSET) = strides_[i];
        simtB64Local(i + SIMT_FUSEDOUTPUTINNERSHAPE_OFFSET) = fusedOutputInnerShape_[i];
    }
    PipeBarrier<PIPE_ALL>();
    asc_vf_call<CopyOut<T>>(dim3(threadsPerBlock_), 
                              (__local_mem__ T*)(yLocal.GetPhyAddr()), (__gm__ T*)(dstGm_.GetPhyAddr()),  
                              (__local_mem__ uint32_t*)(simtB32Local.GetPhyAddr()), (__local_mem__ int64_t*)(simtB64Local.GetPhyAddr()),
                              curLoopBlockNum_, blockIdx_, normalCoreProcessNum_,
                              loopNum_, normalLoopBlockNum_, inputDimNum_);
    dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void CopyOutSimt<T>::InitCopyParams()
{
    normalCopyParams_ = {static_cast<uint16_t>(1), static_cast<uint32_t>(normalLoopBlockNum_ * sizeof(T)),
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    tailCopyParams_ = {static_cast<uint16_t>(1), static_cast<uint32_t>(tailLoopBlockNum_ * sizeof(T)),
                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    padParams_ = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
}

}  // namespace StridedSliceGrad

#endif  // COPY_OUT_SIMT_H