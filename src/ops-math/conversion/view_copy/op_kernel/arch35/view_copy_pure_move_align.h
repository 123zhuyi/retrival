/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_copy_pure_move_align.h
 * \brief
 */

#ifndef VIEW_COPY_PURE_MOVE_ALIGN_H_
#define VIEW_COPY_PURE_MOVE_ALIGN_H_

#include "view_copy_base.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"


namespace ViewCopy {
using namespace AscendC;
using namespace Ops::Base;

constexpr int16_t ALIGN_32_BYTES = 32;
constexpr int16_t PURE_MOVE_ALIGN_ARRAY_SIZE = 4;

template <typename T>
class ViewCopyPureMoveAlign {
public:
    __aicore__ inline ViewCopyPureMoveAlign(const ViewCopyTilingDataPureMoveAlign& tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR dst, GM_ADDR src);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut(const GlobalTensor<T> &dst, int64_t dstOffset);
    __aicore__ inline void CopyIn(const GlobalTensor<T> &src, int64_t srcOffset);
    __aicore__ inline void CalcCopyOffset(int64_t &srcOffset, int64_t &dstOffset, int64_t globalLoopIdx);
    __aicore__ inline void ProcessPerLoop(int64_t globalLoopNum);

private:
    TPipe &pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inOutQueue_;
    const ViewCopyTilingDataPureMoveAlign& tilingData_;
    GlobalTensor<T> dstGlobal_;
    GlobalTensor<T> srcGlobal_;

    int64_t afterSplitAxisSize_ {1};
    int64_t dstLastDim_ {0};
    int64_t ubFactor_ {0};
    int64_t tailTwoAxisBytes_ {0};
    int64_t tailTwoAxis32AlignBytes_ {0};
    int64_t highAxisSize_ {0};
};

template <typename T>
__aicore__ inline void ViewCopyPureMoveAlign<T>::Init(GM_ADDR dst, GM_ADDR src)
{
    dstLastDim_ = tilingData_.ubDim;
    for (int16_t idx = tilingData_.ubDim + 1; idx < PURE_MOVE_ALIGN_ARRAY_SIZE; idx++) {
        if (tilingData_.pureDstSize[idx] == 0) {
            break;
        }
        dstLastDim_ = idx;
        afterSplitAxisSize_ *= tilingData_.pureDstSize[idx];
    }
    if ((dstLastDim_ - tilingData_.ubDim) >= 2) {
        tailTwoAxisBytes_ = tilingData_.pureDstSize[dstLastDim_ - 1] * tilingData_.pureDstSize[dstLastDim_] * sizeof(T);
        tailTwoAxis32AlignBytes_ = CeilAlign(tailTwoAxisBytes_, static_cast<int64_t>(GetUbBlockSize()));
        highAxisSize_ = (dstLastDim_ - tilingData_.ubDim) == 2 ? 1 : tilingData_.pureDstSize[tilingData_.ubDim + 1];
        pipe_.InitBuffer(inOutQueue_, BUFFER_NUM, tilingData_.ubFactor * highAxisSize_ * tailTwoAxis32AlignBytes_);
    } else {
        pipe_.InitBuffer(inOutQueue_, BUFFER_NUM, tilingData_.ubFactor * afterSplitAxisSize_ * sizeof(T));
    }
    dstGlobal_.SetGlobalBuffer((__gm__ T*)(dst + (tilingData_.dstStorageOffset) * sizeof(T)));
    srcGlobal_.SetGlobalBuffer((__gm__ T*)(src + (tilingData_.srcStorageOffset) * sizeof(T)));
}

template <typename T>
__aicore__ inline void ViewCopyPureMoveAlign<T>::CalcCopyOffset(int64_t &srcOffset, int64_t &dstOffset, int64_t globalLoopIdx)
{
    // 计算src的偏移
    int64_t uoTimes = globalLoopIdx / tilingData_.uo;
    int64_t loopIdxInUo = globalLoopIdx - uoTimes * tilingData_.uo;

    ubFactor_ = loopIdxInUo == (tilingData_.uo - 1) ? tilingData_.tailUbFactor : tilingData_.ubFactor;
    srcOffset = (uoTimes * tilingData_.pureDstSize[tilingData_.ubDim] + loopIdxInUo * tilingData_.ubFactor) * afterSplitAxisSize_;

    // 计算dst的偏移
    int64_t axisTotalTimes = globalLoopIdx / tilingData_.uo;
    int64_t splitDimTimes = globalLoopIdx - axisTotalTimes * tilingData_.uo;
    int64_t axisCurrentTimes = axisTotalTimes;
    int64_t axisLeftTimes = 0;

    for (int16_t idx = tilingData_.ubDim - 1; idx >= 0; --idx) {
        axisCurrentTimes = axisTotalTimes / tilingData_.pureDstSize[idx];
        axisLeftTimes = axisTotalTimes - axisCurrentTimes * tilingData_.pureDstSize[idx];
        dstOffset += axisLeftTimes * tilingData_.pureDstStride[idx];
        axisTotalTimes = axisCurrentTimes;
    }
    dstOffset = dstOffset + splitDimTimes * tilingData_.ubFactor * tilingData_.pureDstStride[tilingData_.ubDim];
}

template <typename T>
__aicore__ inline void ViewCopyPureMoveAlign<T>::CopyIn(const GlobalTensor<T> &src, int64_t srcOffset)
{
    LocalTensor<T> srcLocal = inOutQueue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams = {false, 0, 0, 0};
    DataCopyExtParams extParams = {1, 1, 0, 0, 0};
    
    if ((dstLastDim_ - tilingData_.ubDim) >= 2) {
        padExtParams.isPad = true;
        padExtParams.rightPadding = (tailTwoAxis32AlignBytes_ - tailTwoAxisBytes_) / sizeof(T);
        extParams.blockLen = tailTwoAxisBytes_;
        extParams.blockCount = ubFactor_ * highAxisSize_;
        DataCopyPad<T>(srcLocal, src[srcOffset], extParams, padExtParams);
    } else {
        extParams.blockLen = ubFactor_ * afterSplitAxisSize_ * sizeof(T);
        DataCopyPad<T, PaddingMode::Compact>(srcLocal, src[srcOffset], extParams, padExtParams);
    }
    inOutQueue_.EnQue<T>(srcLocal);
}

template <typename T>
__aicore__ inline void ViewCopyPureMoveAlign<T>::CopyOut(const GlobalTensor<T> &dst, int64_t dstOffset)
{
    LocalTensor<T> srcLocal = inOutQueue_.DeQue<T>();
    LoopModeParams loopParams = {1, 1, 0, 0, 0, 0,};
    DataCopyExtParams extParams = {1, 1, 0, 0, 0};

    if ((dstLastDim_ - tilingData_.ubDim) == 3) {
        loopParams.loop1Size = tilingData_.pureDstSize[dstLastDim_ - 2];
        loopParams.loop1DstStride = tilingData_.pureDstStride[dstLastDim_ - 2] * sizeof(T);
        loopParams.loop1SrcStride = tailTwoAxis32AlignBytes_;
        loopParams.loop2Size = ubFactor_;
        loopParams.loop2DstStride = tilingData_.pureDstStride[tilingData_.ubDim] * sizeof(T);
        loopParams.loop2SrcStride = loopParams.loop1SrcStride * loopParams.loop1Size;
        extParams.blockCount = tilingData_.pureDstSize[dstLastDim_ - 1];
        extParams.blockLen = tilingData_.pureDstSize[dstLastDim_] * sizeof(T);
        extParams.dstStride = (tilingData_.pureDstStride[dstLastDim_ - 1] - tilingData_.pureDstSize[dstLastDim_]) * sizeof(T);
    } else if ((dstLastDim_ - tilingData_.ubDim) == 2) {
        loopParams.loop1Size = ubFactor_;
        loopParams.loop1DstStride = tilingData_.pureDstStride[tilingData_.ubDim] * sizeof(T);
        loopParams.loop1SrcStride = tailTwoAxis32AlignBytes_;
        extParams.blockCount = tilingData_.pureDstSize[dstLastDim_ - 1];
        extParams.blockLen = tilingData_.pureDstSize[dstLastDim_] * sizeof(T);
        extParams.dstStride = (tilingData_.pureDstStride[dstLastDim_ - 1] - tilingData_.pureDstSize[dstLastDim_]) * sizeof(T);
    } else if ((dstLastDim_ - tilingData_.ubDim) == 1) {
        int64_t highDimStride0 = tilingData_.pureDstStride[dstLastDim_ - 1] - tilingData_.pureDstSize[dstLastDim_];
        extParams.blockCount = ubFactor_;
        extParams.blockLen = tilingData_.pureDstSize[tilingData_.ubDim + 1] * sizeof(T);
        extParams.dstStride = (tilingData_.pureDstStride[dstLastDim_ - 1] - tilingData_.pureDstSize[dstLastDim_]) * sizeof(T);
    } else {
        extParams.blockLen = ubFactor_ * sizeof(T);
    }

    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad<T, PaddingMode::Compact>(dst[dstOffset], srcLocal, extParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    
    inOutQueue_.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void ViewCopyPureMoveAlign<T>::ProcessPerLoop(int64_t globalLoopIdx)
{
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    CalcCopyOffset(srcOffset, dstOffset, globalLoopIdx);
    CopyIn(srcGlobal_, srcOffset);
    CopyOut(dstGlobal_, dstOffset); 
}

template <typename T>
__aicore__ inline void ViewCopyPureMoveAlign<T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    int64_t blockFactor = GetBlockIdx() == (GetBlockNum() - 1) ? tilingData_.tailBlockFactor : tilingData_.blockFactor;
    for (int64_t loopIdx = 0; loopIdx < blockFactor; loopIdx++) {
        int64_t totalLoopIdx = GetBlockIdx() * tilingData_.blockFactor + loopIdx;
        ProcessPerLoop(totalLoopIdx);
    }
}
} // namespace ViewCopy

#endif  // VIEW_COPY_PURE_MOVE_ALIGN_H_