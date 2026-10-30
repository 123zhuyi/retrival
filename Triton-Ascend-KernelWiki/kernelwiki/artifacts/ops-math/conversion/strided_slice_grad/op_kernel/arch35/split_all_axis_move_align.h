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
 * \file split_all_axis_move_align.h
 * \brief
 */

#ifndef SPLIT_ALL_AXIS_MOVE_ALIGN_H
#define SPLIT_ALL_AXIS_MOVE_ALIGN_H

#include "strided_slice_grad_base.h"

namespace StridedSliceGrad
{
using namespace AscendC;

template <typename T>
class SplitAllAxisMoveAlign : public StridedSliceGradBase
{
public:
    __aicore__ inline SplitAllAxisMoveAlign(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(uint32_t outerAxisIdx, uint32_t outerAxisTail);
    __aicore__ inline void CopyOut(uint32_t outerAxisIdx, uint32_t outerAxisTail);
    __aicore__ inline void InitCopyParams();

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    DataCopyExtParams normalCopyParams_;
    DataCopyExtParams tailCopyParams_;
    DataCopyExtParams curCopyParams_;
    DataCopyPadExtParams<T> padParams_;

    uint32_t curUBLoopCount_;

    uint32_t tailAxisOuter_;
    uint32_t tailAxisInner_;
    uint32_t tailAxisTail_;

    uint32_t blockIdx_;
    uint32_t loopNum_;
};

template <typename T>
__aicore__ inline void SplitAllAxisMoveAlign<T>::Init(GM_ADDR dy, GM_ADDR output,
                                                      const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    // tiling_data
    StridedSliceGradBase::ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    tailAxisOuter_ = tilingData.tailAxisOuter;
    tailAxisInner_ = tilingData.tailAxisInner;
    tailAxisTail_ = tilingData.tailAxisTail;

    // shield normal core and tail core
    curCoreProcessNum_ = (GetBlockIdx() + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;
    curUBLoopCount_ = curCoreProcessNum_;

    InitCopyParams();

    srcGm_.SetGlobalBuffer((__gm__ T*)dy);
    dstGm_.SetGlobalBuffer((__gm__ T*)output);

    pipe_ = pipeIn;
    pipe_.InitBuffer(dataQueue_, DB_BUFFER, bufferSize_);
}

template <typename T>
__aicore__ inline void SplitAllAxisMoveAlign<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (loopNum_ = 0; loopNum_ < curUBLoopCount_; loopNum_++) {
        // sheild normal block and tail block
        curCopyParams_ = ((blockIdx_ * normalCoreProcessNum_ + loopNum_ + 1) % tailAxisOuter_ == 0) ? tailCopyParams_
                                                                                                    : normalCopyParams_;
        uint32_t outerAxisIdx = (blockIdx_ * normalCoreProcessNum_ + loopNum_) / tailAxisOuter_;
        uint32_t outerAxisTail = (blockIdx_ * normalCoreProcessNum_ + loopNum_) % tailAxisOuter_;
        CopyIn(outerAxisIdx, outerAxisTail);
        CopyOut(outerAxisIdx, outerAxisTail);
    }
}

template <typename T>
__aicore__ inline void SplitAllAxisMoveAlign<T>::CopyIn(uint32_t outerAxisIdx, uint32_t outerAxisTail)
{
    uint64_t srcOffset = outerAxisIdx * inputShape_[DIM0] + outerAxisTail * tailAxisInner_;
    LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, srcGm_[srcOffset], curCopyParams_, padParams_);
    dataQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SplitAllAxisMoveAlign<T>::CopyOut(uint32_t outerAxisIdx, uint32_t outerAxisTail)
{
    uint32_t increDim7Idx = outerAxisIdx / fusedSliceNoEndAxis_[DIM7];
    uint32_t increDim7Tail = outerAxisIdx % fusedSliceNoEndAxis_[DIM7];
    uint32_t increDim6Idx = increDim7Tail / fusedSliceNoEndAxis_[DIM6];
    uint32_t increDim6Tail = increDim7Tail % fusedSliceNoEndAxis_[DIM6];
    uint32_t increDim5Idx = increDim6Tail / fusedSliceNoEndAxis_[DIM5];
    uint32_t increDim5Tail = increDim6Tail % fusedSliceNoEndAxis_[DIM5];
    uint32_t increDim4Idx = increDim5Tail / fusedSliceNoEndAxis_[DIM4];
    uint32_t increDim4Tail = increDim5Tail % fusedSliceNoEndAxis_[DIM4];
    uint32_t increDim3Idx = increDim4Tail / fusedSliceNoEndAxis_[DIM3];
    uint32_t increDim3Tail = increDim4Tail % fusedSliceNoEndAxis_[DIM3];
    uint32_t increDim2Idx = increDim3Tail / fusedSliceNoEndAxis_[DIM2];
    uint32_t increDim2Tail = increDim3Tail % fusedSliceNoEndAxis_[DIM2];
    uint32_t increDim1Idx = increDim2Tail;

    uint64_t dstOffset = (begin_[DIM7] + increDim7Idx * strides_[DIM7]) * fusedOutputInnerShape_[DIM7] +
                         (begin_[DIM6] + increDim6Idx * strides_[DIM6]) * fusedOutputInnerShape_[DIM6] +
                         (begin_[DIM5] + increDim5Idx * strides_[DIM5]) * fusedOutputInnerShape_[DIM5] +
                         (begin_[DIM4] + increDim4Idx * strides_[DIM4]) * fusedOutputInnerShape_[DIM4] +
                         (begin_[DIM3] + increDim3Idx * strides_[DIM3]) * fusedOutputInnerShape_[DIM3] +
                         (begin_[DIM2] + increDim2Idx * strides_[DIM2]) * fusedOutputInnerShape_[DIM2] +
                         (begin_[DIM1] + increDim1Idx * strides_[DIM1]) * fusedOutputInnerShape_[DIM1] + begin_[DIM0] +
                         outerAxisTail * tailAxisInner_;
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    DataCopyPad(dstGm_[dstOffset], yLocal, curCopyParams_);
    dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SplitAllAxisMoveAlign<T>::InitCopyParams()
{
    normalCopyParams_ = {static_cast<uint16_t>(1), static_cast<uint32_t>(tailAxisInner_ * sizeof(T)),
                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    tailCopyParams_ = {static_cast<uint16_t>(1), static_cast<uint32_t>(tailAxisTail_ * sizeof(T)),
                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    padParams_ = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
}

}  // namespace StridedSliceGrad

#endif  // SPLIT_ALL_AXIS_MOVE_ALIGN_H