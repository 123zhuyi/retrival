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
 * \file broadcast_to_with_copypad.h
 * \brief kernel of broadcast_to with copy_pad
 */

#ifndef BROADCAST_TO_WITH_DATACOPYPAD_H_
#define BROADCAST_TO_WITH_DATACOPYPAD_H_

#include "broadcast_to_base.h"
#include "kernel_operator.h"

namespace BrcTo {
using namespace AscendC;

constexpr uint8_t bufferNum_datacopypad = 2;
constexpr int32_t queDepth_datacopypad = 1;

template <typename T, typename U>
class BrcToDataCopyPad : public BrcToBase<U> {
public:
    __aicore__ inline BrcToDataCopyPad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyDataIn();
    __aicore__ inline void CopyDataOut();

private:
    const U* tdPtr_;
    int64_t blockIdx = 0;
    AxesLpInfo lpInfo;
    int64_t aBaseIdx = 0;
    int64_t bBaseIdx = 0;
    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, queDepth_datacopypad> que;
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    int64_t gmInOffset;
    int64_t gmOutOffset;
    int64_t inBlockOffset = 0;
    int64_t outBlockOffset = 0;
    AscendC::DataCopyPadExtParams<T> copyInPadParams_{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    uint32_t outLen = sizeof(T);
};

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe_ = pipeIn;
    tdPtr_ = tilingDataPtr;
    pipe_->InitBuffer(que, bufferNum_datacopypad, tdPtr_->tensorSize * sizeof(T));
    blockIdx = GetBlockIdx() % tdPtr_->usedCoreCnt;
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::CopyDataIn()
{
    LocalTensor<T> tensor = que.AllocTensor<T>();
    DataCopyPad(tensor, inGM[gmInOffset], copyParams, copyInPadParams_);
    que.EnQue(tensor);
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::CopyDataOut()
{
    LocalTensor<T> tensor = que.DeQue<T>();
    int64_t bOffset = 0;
    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        DataCopyPad(outGM[gmOutOffset + bOffset], tensor, copyParams);
    }
    que.FreeTensor(tensor);
}

template <typename T, typename U>
__aicore__ inline void BrcToDataCopyPad<T, U>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    this->CalcInBlockOffset(tdPtr_, blockIdx, inBlockOffset);
    this->CalcOutBlockOffset(tdPtr_, blockIdx, outBlockOffset);

    for (int64_t aIdx = 0; aIdx < lpInfo.aLpCnt; aIdx++) {
        copyParams.blockLen = tdPtr_->uLpUnit * outLen;
        int64_t aInOffset = this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx);
        int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx);
        for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
            gmInOffset = (aInOffset + inBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uInOffset * tdPtr_->isUNotB);
            CopyDataIn();
            gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            CopyDataOut();
        }
        if (lpInfo.uLeft > 0) {
            copyParams.blockLen = lpInfo.uLeft * outLen;
            gmInOffset = (aInOffset + inBlockOffset +
                          lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uInOffset * tdPtr_->isUNotB);
            CopyDataIn();
            gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            CopyDataOut();
        }
    }
}

} // namespace BrcTo

#endif // BROADCAST_TO_WITH_DATACOPYPAD_H_
