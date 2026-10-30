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
 * \file split_front_axis_move_align.h
 * \brief
 */

#ifndef SPLIT_FRONT_AXIS_MOVE_ALIGN_H
#define SPLIT_FRONT_AXIS_MOVE_ALIGN_H

#include "strided_slice_grad_base.h"

namespace StridedSliceGrad
{
using namespace AscendC;

template <typename T>
class SplitFrontAxisMoveAlign : public StridedSliceGradBase
{
public:
    __aicore__ inline SplitFrontAxisMoveAlign(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void InitCopyParams();

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    DataCopyExtParams copyOutCopyParams_;
    DataCopyExtParams normalCopyParams_;
    DataCopyExtParams tailCopyParams_;
    DataCopyExtParams curCopyParams_;
    DataCopyPadExtParams<T> padParams_;

    uint32_t curUBLoopCount_;

    uint32_t normalLoopBlockNum_;
    uint32_t tailLoopBlockNum_;
    uint32_t curLoopBlockNum_;
    uint32_t burstLen_;
    uint32_t burstLenAlgin_;
    uint32_t burstNum_;

    uint32_t blockIdx_;
    uint32_t loopNum_;
};

template <typename T>
__aicore__ inline void SplitFrontAxisMoveAlign<T>::Init(GM_ADDR dy, GM_ADDR output,
                                                        const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    // tiling_data
    StridedSliceGradBase::ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    burstLen_ = inputShape_[DIM0];
    burstLenAlgin_ =
        (burstLen_ * bytesForOneData_ + ONE_BLOCK_BYTES - 1) / ONE_BLOCK_BYTES * ONE_BLOCK_BYTES / bytesForOneData_;
    burstNum_ = bufferSize_ / sizeof(T) / burstLenAlgin_;

    // shield normal core and tail core
    curCoreProcessNum_ = (GetBlockIdx() + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    curUBLoopCount_ = (curCoreProcessNum_ + burstNum_ - 1) / burstNum_;

    normalLoopBlockNum_ = (curCoreProcessNum_ >= burstNum_) ? burstNum_ : curCoreProcessNum_;
    tailLoopBlockNum_ = curCoreProcessNum_ - (curUBLoopCount_ - 1) * normalLoopBlockNum_;
    InitCopyParams();

    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_ * burstLen_;
    srcGm_.SetGlobalBuffer((__gm__ T*)dy + intraCoreOffset);
    dstGm_.SetGlobalBuffer((__gm__ T*)output);

    pipe_ = pipeIn;
    pipe_.InitBuffer(dataQueue_, DB_BUFFER, bufferSize_);
}

template <typename T>
__aicore__ inline void SplitFrontAxisMoveAlign<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (loopNum_ = 0; loopNum_ < curUBLoopCount_; loopNum_++) {
        // sheild normal block and tail block
        curLoopBlockNum_ = (loopNum_ == (curUBLoopCount_ - 1)) ? tailLoopBlockNum_ : normalLoopBlockNum_;
        curCopyParams_ = (loopNum_ == (curUBLoopCount_ - 1)) ? tailCopyParams_ : normalCopyParams_;
        CopyIn();
        CopyOut();
    }
}

template <typename T>
__aicore__ inline void SplitFrontAxisMoveAlign<T>::CopyIn()
{
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, srcGm_[normalLoopBlockNum_ * burstLen_ * loopNum_], curCopyParams_, padParams_);
    dataQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SplitFrontAxisMoveAlign<T>::CopyOut()
{
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    for (uint32_t srcIndex = 0; srcIndex < curLoopBlockNum_; srcIndex++) {
        uint32_t absolutelyIdx = blockIdx_ * normalCoreProcessNum_ + loopNum_ * burstNum_ + srcIndex;
        uint32_t incremenDim7Idx = absolutelyIdx / fusedSliceNoEndAxis_[DIM7];
        uint32_t incremenDim7Tail = absolutelyIdx % fusedSliceNoEndAxis_[DIM7];
        uint32_t incremenDim6Idx = incremenDim7Tail / fusedSliceNoEndAxis_[DIM6];
        uint32_t incremenDim6Tail = incremenDim7Tail % fusedSliceNoEndAxis_[DIM6];
        uint32_t incremenDim5Idx = incremenDim6Tail / fusedSliceNoEndAxis_[DIM5];
        uint32_t incremenDim5Tail = incremenDim6Tail % fusedSliceNoEndAxis_[DIM5];
        uint32_t incremenDim4Idx = incremenDim5Tail / fusedSliceNoEndAxis_[DIM4];
        uint32_t incremenDim4Tail = incremenDim5Tail % fusedSliceNoEndAxis_[DIM4];
        uint32_t incremenDim3Idx = incremenDim4Tail / fusedSliceNoEndAxis_[DIM3];
        uint32_t incremenDim3Tail = incremenDim4Tail % fusedSliceNoEndAxis_[DIM3];
        uint32_t incremenDim2Idx = incremenDim3Tail / fusedSliceNoEndAxis_[DIM2];
        uint32_t incremenDim2Tail = incremenDim3Tail % fusedSliceNoEndAxis_[DIM2];
        uint32_t incremenDim1Idx = incremenDim2Tail;

        uint64_t dstOffset = (begin_[DIM7] + incremenDim7Idx * strides_[DIM7]) * fusedOutputInnerShape_[DIM7] +
                             (begin_[DIM6] + incremenDim6Idx * strides_[DIM6]) * fusedOutputInnerShape_[DIM6] +
                             (begin_[DIM5] + incremenDim5Idx * strides_[DIM5]) * fusedOutputInnerShape_[DIM5] +
                             (begin_[DIM4] + incremenDim4Idx * strides_[DIM4]) * fusedOutputInnerShape_[DIM4] +
                             (begin_[DIM3] + incremenDim3Idx * strides_[DIM3]) * fusedOutputInnerShape_[DIM3] +
                             (begin_[DIM2] + incremenDim2Idx * strides_[DIM2]) * fusedOutputInnerShape_[DIM2] +
                             (begin_[DIM1] + incremenDim1Idx * strides_[DIM1]) * fusedOutputInnerShape_[DIM1] +
                             begin_[DIM0];
        DataCopyPad(dstGm_[dstOffset], yLocal[srcIndex * burstLenAlgin_], copyOutCopyParams_);
    }
    dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SplitFrontAxisMoveAlign<T>::InitCopyParams()
{
    normalCopyParams_ = {static_cast<uint16_t>(normalLoopBlockNum_), static_cast<uint32_t>(burstLen_ * sizeof(T)),
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    tailCopyParams_ = {static_cast<uint16_t>(tailLoopBlockNum_), static_cast<uint32_t>(burstLen_ * sizeof(T)),
                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    copyOutCopyParams_ = {static_cast<uint16_t>(1), static_cast<uint32_t>(burstLen_ * sizeof(T)),
                          static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    padParams_ = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
}

}  // namespace StridedSliceGrad

#endif  // SPLIT_FRONT_AXIS_MOVE_ALIGN_H