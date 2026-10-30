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
 * \file broadcast_to_with_dma.h
 * \brief kernel of broadcast_to with nddma
 */

#ifndef BROADCAST_TO_WITH_NDDMA_H_
#define BROADCAST_TO_WITH_NDDMA_H_

#include "broadcast_to_base.h"
#include "kernel_operator.h"

namespace BrcTo {
using namespace AscendC;

constexpr uint8_t bufferNum = 1;
constexpr int32_t queDepth = 1;
constexpr MultiCopyConfig copyCfg{false, 0, 0, false};

template <typename T, typename U, uint8_t maxDim = 4>
class BrcToWithNDDMA : public BrcToBase<U> {
public:
    __aicore__ inline BrcToWithNDDMA(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetDMAParams();
    __aicore__ inline void CalcOutSize();
    __aicore__ inline void CopyDataInWithDMA(LocalTensor<T>& tensor);
    __aicore__ inline void CopyDataOut(LocalTensor<T>& tensor);

private:
    const U* tdPtr_;
    int64_t blockIdx = 0;
    AxesLpInfo lpInfo;
    int64_t aBaseIdx = 0;
    int64_t bBaseIdx = 0;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, queDepth> que;
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    int64_t gmInOffset;
    int64_t gmOutOffset;
    int64_t inBlockOffset = 0;
    int64_t outBlockOffset = 0;
    AscendC::MultiCopyLoopInfo<maxDim> copyLpInfo;
    AscendC::MultiCopyParams<T, maxDim> mCopyParams;
    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    uint32_t outLen = sizeof(T);
    uint8_t copySwitch = 1; // to avoid repeat copy in for U is broadcast axis
};

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe_ = pipeIn;
    tdPtr_ = tilingDataPtr;
    pipe_->InitBuffer(que, bufferNum, tdPtr_->tensorSize * sizeof(T));
    blockIdx = GetBlockIdx() % tdPtr_->usedCoreCnt;
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::SetDMAParams()
{
    for (uint8_t i = 0; i < maxDim; i++) {
        copyLpInfo.loopSrcStride[i] = tdPtr_->xSrcStride[i];
        copyLpInfo.loopDstStride[i] = tdPtr_->xDstStride[i];
        copyLpInfo.loopSize[i] = tdPtr_->xSize[i];
    }
    mCopyParams.loopInfo = copyLpInfo;
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::CopyDataInWithDMA(LocalTensor<T>& tensor)
{
    if (copySwitch > 0) {
        DataCopy<T, maxDim, copyCfg>(tensor, inGM[gmInOffset], mCopyParams);
        this->InsertSync(HardEvent::MTE2_MTE3);
        copySwitch = copySwitch * uint8_t(tdPtr_->isUNotB);
    }
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::CalcOutSize()
{
    for (uint8_t i = 1; i < maxDim; i++) {
        outLen *= tdPtr_->xSize[i];
    }
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::CopyDataOut(LocalTensor<T>& tensor)
{
    int64_t bOffset = 0;
    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        DataCopyPad(outGM[gmOutOffset + bOffset], tensor, copyParams);
    }
    this->InsertSync(HardEvent::MTE3_MTE2);
}

template <typename T, typename U, uint8_t maxDim>
__aicore__ inline void BrcToWithNDDMA<T, U, maxDim>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    this->CalcInBlockOffset(tdPtr_, blockIdx, inBlockOffset);
    this->CalcOutBlockOffset(tdPtr_, blockIdx, outBlockOffset);

    LocalTensor<T> tensor = que.AllocTensor<T>();

    SetDMAParams();
    CalcOutSize();
    for (int64_t aIdx = 0; aIdx < lpInfo.aLpCnt; aIdx++) {
        mCopyParams.loopInfo.loopSize[0] = tdPtr_->uLpUnit;
        copyParams.blockLen = tdPtr_->uLpUnit * outLen;
        copySwitch = 1;
        int64_t aInOffset = this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx);
        int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx);
        for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
            gmInOffset = (aInOffset + inBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uInOffset * tdPtr_->isUNotB);
            CopyDataInWithDMA(tensor);
            gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            CopyDataOut(tensor);
        }
        if (lpInfo.uLeft > 0) {
            mCopyParams.loopInfo.loopSize[0] = lpInfo.uLeft;
            copyParams.blockLen = lpInfo.uLeft * outLen;
            gmInOffset = (aInOffset + inBlockOffset +
                          lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uInOffset * tdPtr_->isUNotB);
            CopyDataInWithDMA(tensor);
            gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            CopyDataOut(tensor);
        }
    }

    que.FreeTensor(tensor);
}

} // namespace BrcTo

#endif // BROADCAST_TO_WITH_NDDMA_H_
