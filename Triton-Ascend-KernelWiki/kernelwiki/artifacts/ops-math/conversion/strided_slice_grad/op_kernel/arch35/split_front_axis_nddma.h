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
 * \file split_front_axis_nddma.h
 * \brief
 */
#ifndef SPLIT_FRONT_AXIS_NDDMA_H
#define SPLIT_FRONT_AXIS_NDDMA_H

#include "strided_slice_grad_base.h"

namespace StridedSliceGrad
{
using namespace AscendC;

template <typename T>
class SplitFrontAxisNddma : public StridedSliceGradBase
{
public:
    __aicore__ inline SplitFrontAxisNddma(){};
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline bool IsNeedNddma();
    __aicore__ inline void ProcessNddma(LocalTensor<T>& srcLocal);
    __aicore__ inline void InitCopyParams();

private:
    __aicore__ inline void ProcessPerLoop(LocalTensor<T>& srcLocal, TEventID eventId);

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> ubBuf1_;
    TBuf<QuePosition::VECCALC> ubBuf2_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    DataCopyExtParams copyOutCopyParams_;

    MultiCopyParams<T, NDDMA_MAX_DIMS> dmaParam_;
    static constexpr MultiCopyConfig nddmaConfig_ = {false};

    uint32_t splitUbAxisNum_;
    uint32_t splitUbAxisIndex_;
    int64_t ubDataSize_;
    int64_t fusedOutputShapeNoInnerUb_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t indexOfNddma_[MAX_SUPPORT_DIM] = {1, 1, 1, 1, 1, 1, 1, 1};

    uint32_t blockIdx_;
    uint32_t loopNum_;
};

template <typename T>
__aicore__ inline void SplitFrontAxisNddma<T>::Init(GM_ADDR dy, GM_ADDR output,
                                                    const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    // tiling_data
    StridedSliceGradBase::ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    splitUbAxisNum_ = tilingData.splitUbAxisNum;  // ub里面能放得下的维度数目  [1,5]
    splitUbAxisIndex_ = MAX_SUPPORT_DIM - 1 - tilingData.splitUbAxisNum;
    ubDataSize_ = fusedOutputInnerShape_[splitUbAxisIndex_];

    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        fusedOutputShapeNoInnerUb_[i] = fusedOutputInnerShape_[i] / ubDataSize_;
    }

    // shield normal core and tail core
    curCoreProcessNum_ = (GetBlockIdx() + 1 == usedCoreNum_) ? tailCoreProcessNum_ : normalCoreProcessNum_;

    InitCopyParams();

    srcGm_.SetGlobalBuffer((__gm__ T*)dy);
    dstGm_.SetGlobalBuffer((__gm__ T*)output);

    pipe_ = pipeIn;
    pipe_.InitBuffer(ubBuf1_, bufferSize_);
    pipe_.InitBuffer(ubBuf2_, bufferSize_);
}

template <typename T>
__aicore__ inline bool SplitFrontAxisNddma<T>::IsNeedNddma()
{
    uint32_t dimIdx = 0;
    uint32_t dimIdxTail = blockIdx_ * normalCoreProcessNum_ + loopNum_;
    for (uint32_t dimNum = 0; dimNum <= splitUbAxisIndex_; dimNum++) {
        dimIdx = dimIdxTail / fusedOutputShapeNoInnerUb_[dimNum];
        dimIdxTail = dimIdxTail % fusedOutputShapeNoInnerUb_[dimNum];
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
__aicore__ inline void SplitFrontAxisNddma<T>::Process()
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
__aicore__ inline void SplitFrontAxisNddma<T>::ProcessNddma(LocalTensor<T>& srcLocal)
{
    int64_t ubOffset = 0;
    for (uint32_t i = MAX_SUPPORT_DIM - 1; i > splitUbAxisIndex_; i--) {
        ubOffset += begin_[i] * fusedOutputInnerShape_[i];
    }

    int64_t srcOffset = 0;
    for (uint32_t i = 0; i <= splitUbAxisIndex_; i++) {
        srcOffset += (indexOfNddma_[i] - begin_[i]) / strides_[i] * fusedSliceInnerShape_[i];
    }
    DataCopy<T, NDDMA_MAX_DIMS, nddmaConfig_>(srcLocal[ubOffset], srcGm_[srcOffset], dmaParam_);
}

template <typename T>
__aicore__ inline void SplitFrontAxisNddma<T>::ProcessPerLoop(LocalTensor<T>& srcLocal, TEventID eventId)
{
    WaitFlag<HardEvent::MTE3_V>(eventId);
    Duplicate(srcLocal, T(0), ubDataSize_);
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);

    if (IsNeedNddma()) {
        ProcessNddma(srcLocal);
    }

    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);

    int64_t dstOffset = (blockIdx_ * normalCoreProcessNum_ + loopNum_) * ubDataSize_;
    DataCopyPad(dstGm_[dstOffset], srcLocal, copyOutCopyParams_);
    SetFlag<HardEvent::MTE3_V>(eventId);
}

template <typename T>
__aicore__ inline void SplitFrontAxisNddma<T>::InitCopyParams()
{
    copyOutCopyParams_ = {static_cast<uint16_t>(1), static_cast<uint32_t>(ubDataSize_ * sizeof(T)),
                          static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    dmaParam_.constantValue = 0;
    for (uint32_t i = 0; i < splitUbAxisNum_; i++) {
        // [低维  --> 高维]
        dmaParam_.loopInfo.loopSize[i] = inputShape_[MAX_SUPPORT_DIM - 1 - i];
        dmaParam_.loopInfo.loopSrcStride[i] = fusedSliceInnerShape_[MAX_SUPPORT_DIM - 1 - i];
        dmaParam_.loopInfo.loopDstStride[i] =
            fusedOutputInnerShape_[MAX_SUPPORT_DIM - 1 - i] * strides_[MAX_SUPPORT_DIM - 1 - i];
        dmaParam_.loopInfo.loopLpSize[i] = 0;
        dmaParam_.loopInfo.loopRpSize[i] = 0;
    }

    // 补维
    for (uint32_t i = splitUbAxisNum_; i < NDDMA_MAX_DIMS; i++) {
        dmaParam_.loopInfo.loopSize[i] = 1;
        dmaParam_.loopInfo.loopSrcStride[i] = 0;
        dmaParam_.loopInfo.loopDstStride[i] = 0;
        dmaParam_.loopInfo.loopLpSize[i] = 0;
        dmaParam_.loopInfo.loopRpSize[i] = 0;
    }
}

}  // namespace StridedSliceGrad

#endif  // SPLIT_FRONT_AXIS_NDDMA_H