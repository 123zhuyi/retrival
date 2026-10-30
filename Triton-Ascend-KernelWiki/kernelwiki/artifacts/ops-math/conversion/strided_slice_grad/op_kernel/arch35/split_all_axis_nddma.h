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
 * \file split_all_axis_nddma.h
 * \brief
 */
#ifndef SPLIT_ALL_AXIS_NDDMA_H
#define SPLIT_ALL_AXIS_NDDMA_H

#include "strided_slice_grad_base.h"

namespace StridedSliceGrad
{
using namespace AscendC;

template <typename T>
class SplitAllAxisNddma : public StridedSliceGradBase
{
public:
    __aicore__ inline SplitAllAxisNddma(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline bool IsNeedNddma();
    __aicore__ inline void ProcessNddma(LocalTensor<T>& srcLocal);
    __aicore__ inline void ProcessPerLoop(LocalTensor<T>& srcLocal, TEventID eventId);

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ubBuf1_;
    TBuf<QuePosition::VECCALC> ubBuf2_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    MultiCopyParams<T, 1> dmaParam_;
    static constexpr MultiCopyConfig nddmaConfig_ = {false};

    int64_t dealDataSize_;

    uint32_t tailAxisOuter_;
    uint32_t tailAxisInner_;
    uint32_t tailAxisTail_;

    int64_t outerAxisIdx_;
    int64_t outerAxisTail_;
    int64_t indexOfNddma_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};

    uint32_t blockIdx_;
    uint32_t loopNum_;
};

template <typename T>
__aicore__ inline void SplitAllAxisNddma<T>::Init(GM_ADDR dy, GM_ADDR output,
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

    srcGm_.SetGlobalBuffer((__gm__ T*)dy);
    dstGm_.SetGlobalBuffer((__gm__ T*)output);

    pipe_ = pipeIn;
    pipe_.InitBuffer(ubBuf1_, bufferSize_);
    pipe_.InitBuffer(ubBuf2_, bufferSize_);
}

template <typename T>
__aicore__ inline bool SplitAllAxisNddma<T>::IsNeedNddma()
{
    if ((outerAxisTail_ + 1) * tailAxisInner_ <= begin_[DIM0] || outerAxisTail_ * tailAxisInner_ >= end_[DIM0]) {
        return false;
    }

    uint32_t dimIdx = 0;
    int64_t dimIdxTail = outerAxisIdx_;
    for (uint32_t dimNum = 0; dimNum <= DIM1; dimNum++) {
        dimIdx = dimIdxTail / fusedOutputNoEndAxis_[dimNum];
        dimIdxTail = dimIdxTail % fusedOutputNoEndAxis_[dimNum];
        if (dimIdx >= begin_[dimNum] && dimIdx < end_[dimNum] && (dimIdx - begin_[dimNum]) % strides_[dimNum] == 0) {
            indexOfNddma_[dimNum] = dimIdx;
            continue;
        } else {
            return false;
        }
    }
    return true;
}

template <typename T>
__aicore__ inline void SplitAllAxisNddma<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    LocalTensor<T> srcLocal1 = ubBuf1_.Get<T>();
    LocalTensor<T> srcLocal2 = ubBuf2_.Get<T>();

    TEventID eventFirst = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    TEventID eventSecond = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(eventFirst);
    SetFlag<HardEvent::MTE3_V>(eventSecond);
    for (loopNum_ = 0; loopNum_ < curCoreProcessNum_; loopNum_++) {
        outerAxisIdx_ = (blockIdx_ * normalCoreProcessNum_ + loopNum_) / tailAxisOuter_;
        outerAxisTail_ =
            (blockIdx_ * normalCoreProcessNum_ + loopNum_) % tailAxisOuter_;  // range [0--> tailAxisOuter_ - 1]

        dealDataSize_ = (outerAxisTail_ == tailAxisOuter_ - 1) ? tailAxisTail_ : tailAxisInner_;

        if (loopNum_ % DB_BUFFER == 0) {
            ProcessPerLoop(srcLocal1, eventFirst);
        } else {
            ProcessPerLoop(srcLocal2, eventSecond);
        }
    }
    WaitFlag<HardEvent::MTE3_V>(eventFirst);
    WaitFlag<HardEvent::MTE3_V>(eventSecond);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventFirst);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventSecond);
}

template <typename T>
__aicore__ inline void SplitAllAxisNddma<T>::ProcessNddma(LocalTensor<T>& srcLocal)
{
    int64_t ubOffset = 0;
    int64_t srcGmEndAxisOffset = 0;
    // ubOffset  srcGmEndAxisOffset 值只跟当前段头和begin的相对关系有关
    if (outerAxisTail_ * tailAxisInner_ > begin_[DIM0]) {
        int64_t temp = (outerAxisTail_ * tailAxisInner_ - begin_[DIM0]) % strides_[DIM0];
        ubOffset = temp == 0 ? 0 : strides_[DIM0] - temp;
        //  首尾包含的元素数量（左闭右开）： (end - 1 - begin)//stride + 1   前面的元素个数就是本ub内开始的元素的索引
        srcGmEndAxisOffset = (outerAxisTail_ * tailAxisInner_ - begin_[DIM0] - 1) / strides_[DIM0] + 1;
    } else {
        ubOffset = begin_[DIM0] - outerAxisTail_ * tailAxisInner_;
        srcGmEndAxisOffset = 0;
    }

    int64_t elementSize = 0;  // 本ubBuf内的元素数量，值只跟当前段尾和end的相对关系有关
    if ((outerAxisTail_ + 1) * tailAxisInner_ < end_[DIM0]) {
        elementSize = (tailAxisInner_ - ubOffset) / strides_[DIM0] + 1;  // 左闭不一定右开 （end-begin）//stride + 1
    } else {
        elementSize = (end_[DIM0] - outerAxisTail_ * tailAxisInner_ - ubOffset - 1) / strides_[DIM0] +
                      1;  // 左闭右开 （end-begin -1）//stride + 1
    }

    dmaParam_.constantValue = 0;
    dmaParam_.loopInfo.loopSize[0] = elementSize;
    dmaParam_.loopInfo.loopSrcStride[0] = 1;
    dmaParam_.loopInfo.loopDstStride[0] = strides_[DIM0];
    dmaParam_.loopInfo.loopLpSize[0] = 0;
    dmaParam_.loopInfo.loopRpSize[0] = 0;

    int64_t srcOffset = 0;
    for (uint32_t i = 0; i <= DIM1; i++) {
        srcOffset += (indexOfNddma_[i] - begin_[i]) / strides_[i] * fusedSliceInnerShape_[i];
    }

    srcOffset += srcGmEndAxisOffset;
    DataCopy<T, 1, nddmaConfig_>(srcLocal[ubOffset], srcGm_[srcOffset], dmaParam_);
}

template <typename T>
__aicore__ inline void SplitAllAxisNddma<T>::ProcessPerLoop(LocalTensor<T>& srcLocal, TEventID eventId)
{
    WaitFlag<HardEvent::MTE3_V>(eventId);
    Duplicate(srcLocal, T(0), dealDataSize_);
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);

    if (IsNeedNddma()) {
        ProcessNddma(srcLocal);
    }

    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);

    int64_t dstOffset = outerAxisIdx_ * outputShape_[DIM0] + outerAxisTail_ * tailAxisInner_;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dealDataSize_ * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(dstGm_[dstOffset], srcLocal, copyParams);
    SetFlag<HardEvent::MTE3_V>(eventId);
}

}  // namespace StridedSliceGrad

#endif  // SPLIT_ALL_AXIS_NDDMA_H