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
 * \file broadcast_to_base.h
 * \brief base class of broadcast_to
 */

#ifndef BROADCAST_TO_BASE_H_
#define BROADCAST_TO_BASE_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace BrcTo {
using namespace AscendC;

constexpr uint8_t aAxisPos = 0;
constexpr uint8_t bAxisPos = 1;
constexpr int64_t uAxisPos = 2;
constexpr uint32_t aParamUnit = 3;
constexpr uint32_t bParamUnit = 2;
constexpr uint32_t nTwo = 2;
constexpr uint32_t nThree = 3;

struct AxesLpInfo {
    int64_t aLpCnt = 1;
    int64_t bLpCnt = 1;
    int64_t uLpCnt = 1;
    int64_t uLeft = 0;
    int64_t uLpBegIdx = 0;
    int64_t uLpEndIdx = 0;
    int64_t bLpBegIdx = 0;
    int64_t bLpEndIdx = 0;
};

template <typename U>
class BrcToBase {
public:
    __aicore__ inline BrcToBase(){};

protected:
    __aicore__ inline void InsertSync(const HardEvent& event);
    __aicore__ inline int64_t CalcAInAxesOffset(const U* tdPtr, int64_t curIdx);
    __aicore__ inline int64_t CalcAOutAxesOffset(const U* tdPtr, int64_t curIdx);
    __aicore__ inline int64_t CalcBAxesOffset(const U* tdPtr, int64_t curIdx);
    __aicore__ inline void CalcInBlockOffset(const U* tdPtr, int64_t blockIdx, int64_t& offset);
    __aicore__ inline void CalcOutBlockOffset(const U* tdPtr, int64_t blockIdx, int64_t& offset);
    __aicore__ inline void CalcABBaseIdx(const U* tdPtr, int64_t blockIdx, int64_t& aBaseIdx, int64_t& bBaseIdx);
    __aicore__ inline void CalcAxesLoopInfo(const U* tdPtr, int64_t blockIdx, AxesLpInfo& lpInfo);
};

template <typename U>
__aicore__ inline void BrcToBase<U>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    switch (event) {
        case HardEvent::V_MTE3:
            SetFlag<HardEvent::V_MTE3>(eventID);
            WaitFlag<HardEvent::V_MTE3>(eventID);
            break;
        case HardEvent::V_MTE2:
            SetFlag<HardEvent::V_MTE2>(eventID);
            WaitFlag<HardEvent::V_MTE2>(eventID);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            break;
        case HardEvent::MTE2_V:
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            break;
        case HardEvent::MTE2_MTE3:
            SetFlag<HardEvent::MTE2_MTE3>(eventID);
            WaitFlag<HardEvent::MTE2_MTE3>(eventID);
            break;
        default:
            break;
    }
}

template <typename U>
__aicore__ inline int64_t BrcToBase<U>::CalcAInAxesOffset(const U* tdPtr, int64_t curIdx)
{
    int64_t offset = 0;
    if (tdPtr->aAxesNum == 0) {
        return offset;
    }
    offset = curIdx * tdPtr->aAxesParams[(tdPtr->aAxesNum - 1) * aParamUnit + 1];
    return offset;
}

template <typename U>
__aicore__ inline int64_t BrcToBase<U>::CalcAOutAxesOffset(const U* tdPtr, int64_t curIdx)
{
    int64_t offset = 0;
    for (int32_t i = tdPtr->aAxesNum - 1; i >= 0; --i) {
        offset += (curIdx % tdPtr->aAxesParams[i * aParamUnit] * tdPtr->aAxesParams[i * aParamUnit + nTwo]);
        curIdx /= tdPtr->aAxesParams[i * aParamUnit];
    }
    return offset;
}

template <typename U>
__aicore__ inline int64_t BrcToBase<U>::CalcBAxesOffset(const U* tdPtr, int64_t curIdx)
{
    int64_t offset = 0;
    for (int32_t i = tdPtr->bAxesNum - 1; i >= 0; --i) {
        offset += (curIdx % tdPtr->bAxesParams[i * bParamUnit] * tdPtr->bAxesParams[i * bParamUnit + 1]);
        curIdx /= tdPtr->bAxesParams[i * bParamUnit];
    }
    return offset;
}

template <typename U>
__aicore__ inline void BrcToBase<U>::CalcInBlockOffset(const U* tdPtr, int64_t blockIdx, int64_t& offset)
{
    if (tdPtr->blockAxis == uAxisPos) {
        offset = blockIdx * tdPtr->ntcULen * tdPtr->uInOffset * tdPtr->isUNotB;
        return;
    }
    offset = 0;
}

template <typename U>
__aicore__ inline void BrcToBase<U>::CalcOutBlockOffset(const U* tdPtr, int64_t blockIdx, int64_t& offset)
{
    if (tdPtr->blockAxis == uAxisPos) {
        offset = blockIdx * tdPtr->ntcULen * tdPtr->uOutOffset;
        return;
    }
    offset = 0;
}

template <typename U>
__aicore__ inline void BrcToBase<U>::CalcAxesLoopInfo(const U* tdPtr, int64_t blockIdx, AxesLpInfo& lpInfo)
{
    if (blockIdx != tdPtr->usedCoreCnt - 1) {
        lpInfo.aLpCnt = tdPtr->ntcALen;
        lpInfo.bLpCnt = tdPtr->ntcBLen;
        lpInfo.uLpCnt = tdPtr->ntcULen / tdPtr->uLpUnit;
        lpInfo.uLeft = tdPtr->ntcULen % tdPtr->uLpUnit;
    } else {
        lpInfo.aLpCnt = tdPtr->tcALen;
        lpInfo.bLpCnt = tdPtr->tcBLen;
        lpInfo.uLpCnt = tdPtr->tcULen / tdPtr->uLpUnit;
        lpInfo.uLeft = tdPtr->tcULen % tdPtr->uLpUnit;
    }

    int64_t realBlockIdx = GetBlockIdx();
    int64_t blkFactor = realBlockIdx / tdPtr->usedCoreCnt;
    if (tdPtr->doubleMode == 1U) {
        int64_t bCntPerCore = Ops::Base::CeilDiv(lpInfo.bLpCnt, tdPtr->dFactor);
        if (blkFactor < tdPtr->dFactor - 1) {
            lpInfo.bLpBegIdx = bCntPerCore * blkFactor;
            lpInfo.bLpEndIdx = bCntPerCore * (blkFactor + 1);
        } else {
            lpInfo.bLpBegIdx = bCntPerCore * blkFactor;
            lpInfo.bLpEndIdx = lpInfo.bLpCnt;
        }
        lpInfo.uLpBegIdx = 0;
        lpInfo.uLpEndIdx = lpInfo.uLpCnt;
        return;
    }
    if (tdPtr->doubleMode == nTwo) {
        int64_t uCntPerCore = Ops::Base::CeilDiv(lpInfo.uLpCnt, tdPtr->dFactor);
        if (blkFactor < tdPtr->dFactor - 1) {
            lpInfo.uLpBegIdx = uCntPerCore * blkFactor;
            lpInfo.uLpEndIdx = uCntPerCore * (blkFactor + 1);
            lpInfo.uLeft = 0;
        } else {
            lpInfo.uLpBegIdx = uCntPerCore * blkFactor;
            lpInfo.uLpEndIdx = lpInfo.uLpCnt;
            lpInfo.uLeft = tdPtr->tcULen % tdPtr->uLpUnit;
        }
        lpInfo.bLpBegIdx = 0;
        lpInfo.bLpEndIdx = lpInfo.bLpCnt;
        return;
    }
    lpInfo.uLpBegIdx = 0;
    lpInfo.uLpEndIdx = lpInfo.uLpCnt;
    lpInfo.bLpBegIdx = 0;
    lpInfo.bLpEndIdx = lpInfo.bLpCnt;
}

template <typename U>
__aicore__ inline void BrcToBase<U>::CalcABBaseIdx(const U* tdPtr, int64_t blockIdx, int64_t& aBaseIdx,
                                                   int64_t& bBaseIdx)
{
    if (tdPtr->blockAxis == aAxisPos) {
        aBaseIdx = blockIdx * tdPtr->ntcALen;
        return;
    }
    if (tdPtr->blockAxis == bAxisPos) {
        bBaseIdx = blockIdx * tdPtr->ntcBLen;
    }
}

} // namespace BrcTo

#endif // BROADCAST_TO_BASE_H_
