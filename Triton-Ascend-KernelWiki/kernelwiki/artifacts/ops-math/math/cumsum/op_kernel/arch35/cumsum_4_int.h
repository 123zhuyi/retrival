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
 * \file cumsum_4_int.h
 * \brief cumsum for dtype int32 and int64
 */

#ifndef CUMSUM_4_INT_
#define CUMSUM_4_INT_

#include <type_traits>

#include "op_kernel/math_util.h"
#include "kernel_operator.h"

namespace Cum {
using namespace AscendC;
constexpr int64_t MAX_STRIDE = 0xffffffff;
constexpr uint32_t MAX_REPEAT = 0xfff;

struct CopyPara {
    int64_t blockCount = 0;
    uint32_t blockLen = 0;
    int64_t gmGap = 0;
    uint32_t ubGap = 0;
    uint32_t elemPerBlock = 0;
};

template <typename T>
class CumTypeBase {
public:
    using IdxType = std::conditional_t<sizeof(T) == 1, uint16_t, uint32_t>;
    using RangeType = std::conditional_t<sizeof(T) == 1, int16_t, int32_t>;
    using VLType = std::conditional_t<sizeof(T) == 1, uint16_t, T>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
};

template <typename T>
static __aicore__ inline void SetUBZero(const LocalTensor<T>& ubTensor, uint32_t len)
{
    static constexpr uint32_t regElem = GetVecLen() / sizeof(T);
    uint16_t lpCnt = Ops::Base::CeilDiv(len, regElem);
    __local_mem__ T* yPtr = (__local_mem__ T*)ubTensor.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::MaskReg mask;
        Reg::RegTensor<T> zeros;
        Reg::Duplicate(zeros, (T)0);
        for (uint16_t i = 0; i < lpCnt; i++) {
            mask = Reg::UpdateMask<T>(len);
            Reg::DataCopy<T>(yPtr + i * regElem, zeros, mask);
        }
    }
}

static __aicore__ inline void InsertSync(const HardEvent& event)
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
        case HardEvent::MTE3_V:
            SetFlag<HardEvent::MTE3_V>(eventID);
            WaitFlag<HardEvent::MTE3_V>(eventID);
            break;
        default:
            break;
    }
}

template <typename T>
__aicore__ inline void CumCopyIn(const LocalTensor<T>& inUB, const GlobalTensor<T>& inGM, CopyPara& copyPara)
{
    DataCopyExtParams copyParams{1, uint32_t(copyPara.blockLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    uint32_t lenBA = Ops::Base::CeilAlign(copyPara.blockLen, copyPara.elemPerBlock);
    int64_t srcStride = (copyPara.gmGap - copyPara.blockLen) * sizeof(T);
    uint32_t dstStride = (copyPara.ubGap - lenBA) / copyPara.elemPerBlock;
    if (copyPara.blockCount > MAX_REPEAT || srcStride > MAX_STRIDE) {
        for (int64_t i = 0; i < copyPara.blockCount; i++) {
            DataCopyPad(inUB[i * copyPara.ubGap], inGM[i * copyPara.gmGap], copyParams, padParams);
        }
        return;
    }
    copyParams.blockCount = (uint16_t)copyPara.blockCount;
    copyParams.srcStride = (uint32_t)srcStride;
    copyParams.dstStride = dstStride;
    DataCopyPad(inUB, inGM, copyParams, padParams);
}

template <typename T>
__aicore__ inline void CumCopyOut(const GlobalTensor<T>& outGM, const LocalTensor<T>& outUB, CopyPara& copyPara)
{
    DataCopyExtParams copyParams{1, uint32_t(copyPara.blockLen * sizeof(T)), 0, 0, 0};
    uint32_t lenBA = Ops::Base::CeilAlign(copyPara.blockLen, copyPara.elemPerBlock);
    int64_t dstStride = (copyPara.gmGap - copyPara.blockLen) * sizeof(T);
    uint32_t srcStride = (copyPara.ubGap - lenBA) / copyPara.elemPerBlock;
    if (copyPara.blockCount > MAX_REPEAT || dstStride > MAX_STRIDE) {
        for (int64_t i = 0; i < copyPara.blockCount; i++) {
            DataCopyPad(outGM[i * copyPara.gmGap], outUB[i * copyPara.ubGap], copyParams);
        }
        return;
    }
    copyParams.blockCount = (uint16_t)copyPara.blockCount;
    copyParams.srcStride = srcStride;
    copyParams.dstStride = (uint32_t)dstStride;
    DataCopyPad(outGM, outUB, copyParams);
}

struct ExclusiveParams {
    int64_t excFactor = 0;  // control other loops input offset
    int64_t excRFactor = 0; // control first loop input offset
    bool isExc = false;
};

static __aicore__ inline void GetExclusiveInfo(const Cum4IntTilingData* tilingPtr, uint32_t curBlockIdx,
                                               ExclusiveParams& excParams, uint32_t& firstRLen, int64_t& rLpCnt,
                                               uint32_t& rLeft)
{
    if (tilingPtr->isExclusive) {
        excParams.excFactor = 1;
        if (!tilingPtr->isRBlockAxis) {
            firstRLen = tilingPtr->rLpUnit - 1;
            excParams.isExc = true;
        } else {
            if (tilingPtr->isReverse && curBlockIdx == tilingPtr->usedCoreCnt - 1) {
                if (rLpCnt > 0) {
                    firstRLen = tilingPtr->rLpUnit - 1;
                } else {
                    firstRLen = rLeft - 1;
                    rLpCnt = 1;
                    rLeft = 0;
                }
                excParams.isExc = true;
            } else if (!tilingPtr->isReverse && curBlockIdx == 0) {
                firstRLen = tilingPtr->rLpUnit - 1;
                excParams.isExc = true;
            } else {
                firstRLen = tilingPtr->rLpUnit;
                excParams.excRFactor = 1;
            }
        }
    }
}

constexpr uint32_t regBlock = 8;
constexpr uint8_t bufferNum = 1;
constexpr int32_t queDepth = 1;
constexpr uint32_t nTwo = 2;

template <typename T>
class DataCopyTDRightACum {
public:
    __aicore__ inline DataCopyTDRightACum(const Cum4IntTilingData* tilingData, uint32_t blockIdx)
        : tilingPtr(tilingData), curBlockIdx(blockIdx){};
    __aicore__ inline void CopyTDRightAProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                               const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal);

private:
    __aicore__ inline void GetAxisLoopInfo();
    __aicore__ inline void InnerProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                        const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal, uint32_t rLen,
                                        uint32_t raLen, bool isExc = false);
    __aicore__ inline void CumCopyTDRightA(const LocalTensor<T>& inUB, const LocalTensor<T>& outUB, uint32_t rLen,
                                           uint32_t rightALen, bool isExc = false);
    __aicore__ inline void ForwardProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                          const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                          int64_t laLpIdx, int64_t raLpIdx, int64_t raSize);
    __aicore__ inline void ReverseProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                          const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                          int64_t laLpIdx, int64_t raLpIdx, int64_t raSize);

private:
    static constexpr uint32_t regElem = GetVecLen() / sizeof(T);
    static constexpr uint32_t blockElem = regElem / regBlock;
    int64_t laLpCnt = 0;
    int64_t rLpCnt = 0;
    int64_t raLpCnt = 0;
    uint32_t rLeft = 0;
    uint32_t raLeft = 0;
    uint32_t curBlockIdx = 0;
    int64_t gmOffset = 0;
    int64_t gmOutOffset = 0;
    int64_t totalRSize = 0;
    int32_t ubOffset = 0;
    const Cum4IntTilingData* tilingPtr = nullptr;
    uint32_t rSize = tilingPtr->rLpUnit;
    uint32_t firstRLen = rSize;
    uint32_t raLenBA = Ops::Base::CeilAlign((uint32_t)tilingPtr->raLpUnit, blockElem);
    ExclusiveParams excParams;
};

template <typename T>
__aicore__ inline void DataCopyTDRightACum<T>::GetAxisLoopInfo()
{
    if (curBlockIdx != tilingPtr->usedCoreCnt - 1) {
        laLpCnt = tilingPtr->ntcLALen;
        rLpCnt = tilingPtr->ntcRLen / tilingPtr->rLpUnit;
        rLeft = tilingPtr->ntcRLen % tilingPtr->rLpUnit;
        raLpCnt = tilingPtr->ntcRALen / tilingPtr->raLpUnit;
        raLeft = tilingPtr->ntcRALen % tilingPtr->raLpUnit;
        totalRSize = tilingPtr->ntcRLen;
    } else {
        laLpCnt = tilingPtr->tcLALen;
        rLpCnt = tilingPtr->tcRLen / tilingPtr->rLpUnit;
        rLeft = tilingPtr->tcRLen % tilingPtr->rLpUnit;
        raLpCnt = tilingPtr->tcRALen / tilingPtr->raLpUnit;
        raLeft = tilingPtr->tcRALen % tilingPtr->raLpUnit;
        totalRSize = tilingPtr->tcRLen;
    }
}

template <typename T>
__aicore__ inline void DataCopyTDRightACum<T>::CumCopyTDRightA(const LocalTensor<T>& inUB, const LocalTensor<T>& outUB,
                                                               uint32_t rLen, uint32_t rightALen, bool isExc)
{
    /*  1. The input data layout in UB as follow, support copy data A one time.
     *           |<---------- rOffset ------->|
     *           |<------ rightALen ----->|
     *           |-------------------------xxx|
     *      rLen |-------------------------xxx|
     *           |-------------------------xxx|
     *  2. The position of result of last loop is at R.
     */
    uint16_t rLpCnt = (uint16_t)rLen;
    uint16_t aLpCnt = (uint16_t)Ops::Base::CeilDiv(rightALen, regElem);
    __local_mem__ T* srcPtr = (__local_mem__ T*)inUB.GetPhyAddr();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outUB.GetPhyAddr();

    int32_t bakBegAddr = (rSize - 1) * raLenBA;
    int32_t srcBegAddr = 0;
    int32_t rOffset = raLenBA;
    if (tilingPtr->isReverse) {
        bakBegAddr = 0;
        srcBegAddr = (rLen - 1) * raLenBA;
        rOffset *= -1;
    }
    int32_t dstBegAddr = srcBegAddr;
    if (isExc && !tilingPtr->isReverse) {
        dstBegAddr += rOffset;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::MaskReg mask;
        int32_t offset = 0;
        int32_t offset1 = 0;
        for (uint16_t aIdx = 0; aIdx < aLpCnt; aIdx++) {
            mask = Reg::UpdateMask<T>(rightALen);
            offset1 = aIdx * regElem;
            // Get last loop backup result, the result is zero for first loop
            Reg::DataCopy<T>(yDst, dstPtr + bakBegAddr + offset1);
            for (uint16_t rIdx = 0; rIdx < rLpCnt; rIdx++) {
                offset = offset1 + rIdx * rOffset;
                Reg::DataCopy<T>(xSrc, srcPtr + srcBegAddr + offset);
                Reg::Add(yDst, yDst, xSrc, mask);
                Reg::DataCopy<T>(dstPtr + dstBegAddr + offset, yDst, mask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void DataCopyTDRightACum<T>::InnerProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                            const LocalTensor<T>& inLocal,
                                                            const LocalTensor<T>& outLocal, uint32_t rLen,
                                                            uint32_t raLen, bool isExc)
{
    if (rLen > 0) {
        CopyPara copyPara = {rLen, raLen, tilingPtr->rOffset, raLenBA, blockElem};
        CumCopyIn(inLocal, inGM[gmOffset], copyPara);
        InsertSync(HardEvent::MTE2_V);
        CumCopyTDRightA(inLocal, outLocal, rLen, raLen, isExc);
        InsertSync(HardEvent::V_MTE2);
    }

    InsertSync(HardEvent::V_MTE3);
    if (isExc) {
        rLen += 1;
    }
    CopyPara copyPara = {rLen, raLen, tilingPtr->rOffset, raLenBA, blockElem};
    CumCopyOut(outGM[gmOutOffset], outLocal, copyPara);
    InsertSync(HardEvent::MTE3_V);
}

template <typename T>
__aicore__ inline void DataCopyTDRightACum<T>::ForwardProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                              const LocalTensor<T>& inLocal,
                                                              const LocalTensor<T>& outLocal, int64_t laLpIdx,
                                                              int64_t raLpIdx, int64_t raSize)
{
    SetUBZero(outLocal, rSize * raLenBA);
    InsertSync(HardEvent::V_MTE2);
    if (rLpCnt > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset + raLpIdx * tilingPtr->raLpUnit +
                    curBlockIdx * tilingPtr->blockOffset - excParams.excRFactor * tilingPtr->rOffset);
        gmOutOffset = gmOffset + excParams.excRFactor * tilingPtr->rOffset;
        InnerProcess(inGM, outGM, inLocal, outLocal, firstRLen, raSize, excParams.isExc);
    }
    for (int64_t rLpIdx = 1; rLpIdx < rLpCnt; rLpIdx++) {
        gmOffset = (laLpIdx * tilingPtr->laOffset + raLpIdx * tilingPtr->raLpUnit +
                    curBlockIdx * tilingPtr->blockOffset + (rLpIdx * rSize - excParams.excFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset + excParams.excFactor * tilingPtr->rOffset;
        InnerProcess(inGM, outGM, inLocal, outLocal, rSize, raSize);
    }
    if (rLeft > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset + raLpIdx * tilingPtr->raLpUnit +
                    curBlockIdx * tilingPtr->blockOffset + (rLpCnt * rSize - excParams.excFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset + excParams.excFactor * tilingPtr->rOffset;
        InnerProcess(inGM, outGM, inLocal, outLocal, rLeft, raSize);
    }
}

template <typename T>
__aicore__ inline void DataCopyTDRightACum<T>::ReverseProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                              const LocalTensor<T>& inLocal,
                                                              const LocalTensor<T>& outLocal, int64_t laLpIdx,
                                                              int64_t raLpIdx, int64_t raSize)
{
    SetUBZero(outLocal, rSize * raLenBA);
    InsertSync(HardEvent::V_MTE2);
    if (rLpCnt > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset + raLpIdx * tilingPtr->raLpUnit +
                    curBlockIdx * tilingPtr->blockOffset +
                    (totalRSize - firstRLen + excParams.excRFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
        InnerProcess(inGM, outGM, inLocal, outLocal, firstRLen, raSize, excParams.isExc);
    }
    for (int64_t rLpIdx = rLpCnt - 1; rLpIdx > 0; rLpIdx--) {
        gmOffset = (laLpIdx * tilingPtr->laOffset + raLpIdx * tilingPtr->raLpUnit +
                    curBlockIdx * tilingPtr->blockOffset +
                    (rLpIdx * rSize + rLeft - rSize + excParams.excFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
        InnerProcess(inGM, outGM, inLocal, outLocal, rSize, raSize);
    }
    if (rLeft > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset + raLpIdx * tilingPtr->raLpUnit +
                    curBlockIdx * tilingPtr->blockOffset + excParams.excFactor * tilingPtr->rOffset);
        gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
        InnerProcess(inGM, outGM, inLocal, outLocal, rLeft, raSize);
    }
}

template <typename T>
__aicore__ inline void DataCopyTDRightACum<T>::CopyTDRightAProcess(const GlobalTensor<T>& inGM,
                                                                   const GlobalTensor<T>& outGM,
                                                                   const LocalTensor<T>& inLocal,
                                                                   const LocalTensor<T>& outLocal)
{
    GetAxisLoopInfo();
    GetExclusiveInfo(tilingPtr, curBlockIdx, excParams, firstRLen, rLpCnt, rLeft);

    if (!tilingPtr->isReverse) {
        for (int64_t laLpIdx = 0; laLpIdx < laLpCnt; laLpIdx++) {
            for (int64_t raLpIdx = 0; raLpIdx < raLpCnt; raLpIdx++) {
                ForwardProcess(inGM, outGM, inLocal, outLocal, laLpIdx, raLpIdx, tilingPtr->raLpUnit);
            }
            if (raLeft > 0) {
                ForwardProcess(inGM, outGM, inLocal, outLocal, laLpIdx, raLpCnt, raLeft);
            }
        }
    } else {
        for (int64_t laLpIdx = 0; laLpIdx < laLpCnt; laLpIdx++) {
            for (int64_t raLpIdx = 0; raLpIdx < raLpCnt; raLpIdx++) {
                ReverseProcess(inGM, outGM, inLocal, outLocal, laLpIdx, raLpIdx, tilingPtr->raLpUnit);
            }
            if (raLeft > 0) {
                ReverseProcess(inGM, outGM, inLocal, outLocal, laLpIdx, raLpCnt, raLeft);
            }
        }
    }
}

template <typename T>
class GatherCum : public CumTypeBase<T> {
public:
    __aicore__ inline GatherCum(const Cum4IntTilingData* tilingData, TPipe* inPipe, uint32_t blockIdx)
        : tilingPtr(tilingData), pipe(inPipe), curBlockIdx(blockIdx)
    {
        pipe->InitBuffer(idxBuf, regElem * sizeof(T) * 2);
        idxUB = idxBuf.Get<int32_t>();
    };
    __aicore__ inline void CumGatherTDRightA(const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                             uint32_t rLen, bool isExc = false);
    __aicore__ inline void GatherRightATDLeftAProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                      const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal);
    __aicore__ inline void CumGatherRightATDR(const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                              uint32_t rLen, bool isExc = false);
    __aicore__ inline void NoSplitRInnerProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                int64_t rLen, bool isExc,
                                                void (GatherCum<T>::*impl)(const LocalTensor<T>&, const LocalTensor<T>&,
                                                                           uint32_t, bool));
    __aicore__ inline void GatherNoSplitRProcess(
        const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM, const LocalTensor<T>& inLocal,
        const LocalTensor<T>& outLocal,
        void (GatherCum<T>::*func)(const GlobalTensor<T>&, const GlobalTensor<T>&, const LocalTensor<T>&,
                                   const LocalTensor<T>&, int64_t, bool,
                                   void (GatherCum<T>::*impl)(const LocalTensor<T>&, const LocalTensor<T>&, uint32_t,
                                                              bool)),
        void (GatherCum<T>::*impl)(const LocalTensor<T>&, const LocalTensor<T>&, uint32_t, bool));
    __aicore__ inline void Process(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                   const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal);
    __aicore__ inline void KCheckBGC(uint32_t comCnt, uint32_t arSize);

private:
    __aicore__ inline void GetAxisLoopInfo();
    __aicore__ inline void CumGatherRightATDLeftA(const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                  uint32_t leftALen, uint32_t rLen, uint32_t laOffset,
                                                  bool isExc = false);
    __aicore__ inline void TDLeftAInnerProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                               const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                               int64_t laLen, int64_t rLen, bool isExc = false);
    __aicore__ inline void CleanInputDirtyData(__local_mem__ T*& srcPtr, int32_t zeroBase, uint32_t leftSize);
    __aicore__ inline void TDLeftAForwardProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                 const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                 int64_t laLpIdx, int64_t laSize);
    __aicore__ inline void TDLeftAReverseProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                 const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                 int64_t laLpIdx, int64_t laSize);
    __aicore__ inline void TDRReverseProcess(__local_mem__ T*& srcPtr, __local_mem__ T*& dstPtr, uint16_t rLpNum,
                                             int32_t comRightA, int32_t splitRASize, int32_t curBegAddrBase,
                                             int32_t curBakBegAdd, uint16_t sndRLpCnt, int32_t phs1SrcBegIdx,
                                             int32_t phs2SrcBegIdx, int32_t phs1LpOffset, int32_t phs2LpOffset,
                                             int32_t phs2TLpOffset, int32_t bakBegAddr, int32_t phs1DstBegIdx,
                                             int32_t phs2DstBegIdx, uint32_t comLpMaskVal, uint32_t sndLpLeftMaskVal);
    __aicore__ inline void TDRForwardProcess(__local_mem__ T*& srcPtr, __local_mem__ T*& dstPtr, uint16_t rLpNum,
                                             int32_t comRightA, int32_t splitRASize, int32_t curBegAddrBase,
                                             int32_t curBakBegAdd, uint16_t sndRLpCnt, int32_t phs1SrcBegIdx,
                                             int32_t phs2SrcBegIdx, int32_t phs1LpOffset, int32_t phs2LpOffset,
                                             int32_t phs2TLpOffset, int32_t bakBegAddr, int32_t phs1DstBegIdx,
                                             int32_t phs2DstBegIdx, uint32_t comLpMaskVal, uint32_t sndLpLeftMaskVal);

private:
    using IdxType = typename CumTypeBase<T>::IdxType;
    using RangeType = typename CumTypeBase<T>::RangeType;
    using VLType = typename CumTypeBase<T>::VLType;
    using CastType = typename CumTypeBase<T>::CastType;
    static constexpr uint32_t regElem = GetVecLen() / sizeof(VLType); // used for VF
    static constexpr uint32_t halfRegElem = regElem / 2;
    static constexpr uint32_t blockElem = GetVecLen() / sizeof(T) / regBlock;
    int64_t laLpCnt = 0;
    int64_t rLpCnt = 0;
    uint32_t laLeft = 0;
    uint32_t rLeft = 0;
    uint32_t curBlockIdx = 0;
    int64_t totalRSize = 0;
    int64_t gmOffset = 0;
    int64_t gmOutOffset = 0;
    int32_t ubOffset = 0;
    const Cum4IntTilingData* tilingPtr = nullptr;
    uint32_t rSize = tilingPtr->rLpUnit;
    uint32_t firstRLen = rSize;
    uint32_t arLenBA = Ops::Base::CeilAlign((uint32_t)(tilingPtr->raLpUnit * rSize), blockElem);
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> idxBuf;
    LocalTensor<int32_t> idxUB;
    ExclusiveParams excParams;
    bool isBGC = false;
};

template <typename T>
__aicore__ inline void GatherCum<T>::Process(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                             const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal)
{
    if (tilingPtr->rOffset > halfRegElem ||
        (regElem / tilingPtr->rOffset >= tilingPtr->rLpUnit && tilingPtr->laLpUnit == 1)) {
        GatherNoSplitRProcess(inGM, outGM, inLocal, outLocal, &GatherCum<T>::NoSplitRInnerProcess,
                              &GatherCum<T>::CumGatherTDRightA);
    } else if (tilingPtr->laLpUnit > 1) {
        GatherRightATDLeftAProcess(inGM, outGM, inLocal, outLocal);
    } else {
        GatherNoSplitRProcess(inGM, outGM, inLocal, outLocal, &GatherCum<T>::NoSplitRInnerProcess,
                              &GatherCum<T>::CumGatherRightATDR);
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::CumGatherTDRightA(const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                       uint32_t rLen, bool isExc)
{
    /*  1. The input data layout in UB as follow, support copy data RA one time.
     *           |<--  rightALen -->|
     *           |------------------|
     *      rLen |------------------|
     *           |------------------|
     *  2. The position of result of last loop is at R.
     */

    __local_mem__ T* srcPtr = (__local_mem__ T*)inLocal.GetPhyAddr();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outLocal.GetPhyAddr();
    uint16_t rLpCnt = (uint16_t)rLen;
    uint32_t aSize = tilingPtr->raLpUnit;
    // B8时Gather/Scatter中按照B16处理，mask控制RegTensor中的数据量，Index Tensor控制UB中的数据量，所以B8时mask需要翻倍
    if constexpr (sizeof(T) == 1) {
        aSize += aSize;
    }

    int32_t bakBegAddr = (rSize - 1) * tilingPtr->raLpUnit;
    int32_t srcBegIdx = 0;
    int32_t lpOffset = tilingPtr->raLpUnit;
    if (tilingPtr->isReverse) {
        bakBegAddr = 0;
        srcBegIdx = (rLen - 1) * tilingPtr->raLpUnit;
        lpOffset *= -1;
    }
    int32_t dstBegIdx = srcBegIdx;
    if (isExc && !tilingPtr->isReverse) {
        dstBegIdx += lpOffset;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::RegTensor<RangeType> gIdx;
        Reg::RegTensor<RangeType> sIdx;
        Reg::MaskReg idxMask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
        Reg::MaskReg mask = Reg::UpdateMask<T>(aSize);

        // get backup result of last loop
        Reg::Arange(gIdx, bakBegAddr);
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, mask);

        Reg::Arange(gIdx, srcBegIdx);
        Reg::Arange(sIdx, dstBegIdx);
        for (uint16_t rIdx = 0; rIdx < rLpCnt; rIdx++) {
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, mask);
            Reg::Add(yDst, yDst, xSrc, mask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, mask);
            Reg::Adds(gIdx, gIdx, lpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, lpOffset, idxMask);
        }
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::GetAxisLoopInfo()
{
    if (curBlockIdx != tilingPtr->usedCoreCnt - 1) {
        laLpCnt = tilingPtr->ntcLALen / tilingPtr->laLpUnit;
        laLeft = tilingPtr->ntcLALen % tilingPtr->laLpUnit;
        rLpCnt = tilingPtr->ntcRLen / tilingPtr->rLpUnit;
        rLeft = tilingPtr->ntcRLen % tilingPtr->rLpUnit;
        totalRSize = tilingPtr->ntcRLen;
    } else {
        laLpCnt = tilingPtr->tcLALen / tilingPtr->laLpUnit;
        laLeft = tilingPtr->tcLALen % tilingPtr->laLpUnit;
        rLpCnt = tilingPtr->tcRLen / tilingPtr->rLpUnit;
        rLeft = tilingPtr->tcRLen % tilingPtr->rLpUnit;
        totalRSize = tilingPtr->tcRLen;
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::NoSplitRInnerProcess(
    const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM, const LocalTensor<T>& inLocal,
    const LocalTensor<T>& outLocal, int64_t rLen, bool isExc,
    void (GatherCum<T>::*impl)(const LocalTensor<T>&, const LocalTensor<T>&, uint32_t, bool))
{
    // exclusive is true and rLen is zero, the outputs are zeros
    if (rLen > 0) {
        DataCopyPad(inLocal, inGM[gmOffset], copyParams, padParams);
        InsertSync(HardEvent::MTE2_V);
        (this->*impl)(inLocal, outLocal, rLen, isExc);
        InsertSync(HardEvent::V_MTE2);
    }

    InsertSync(HardEvent::V_MTE3);
    if (isExc) {
        copyParams.blockLen = (uint32_t)(tilingPtr->raLpUnit * (rLen + 1) * sizeof(T));
    }
    DataCopyPad(outGM[gmOutOffset], outLocal, copyParams);
    InsertSync(HardEvent::MTE3_V);
}

template <typename T>
__aicore__ inline void GatherCum<T>::GatherNoSplitRProcess(
    const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM, const LocalTensor<T>& inLocal,
    const LocalTensor<T>& outLocal,
    void (GatherCum<T>::*func)(const GlobalTensor<T>&, const GlobalTensor<T>&, const LocalTensor<T>&,
                               const LocalTensor<T>&, int64_t, bool,
                               void (GatherCum<T>::*impl)(const LocalTensor<T>&, const LocalTensor<T>&, uint32_t,
                                                          bool)),
    void (GatherCum<T>::*impl)(const LocalTensor<T>&, const LocalTensor<T>&, uint32_t, bool))
{
    GetAxisLoopInfo();
    GetExclusiveInfo(tilingPtr, curBlockIdx, excParams, firstRLen, rLpCnt, rLeft);

    if (!tilingPtr->isReverse) {
        for (int64_t laLpIdx = 0; laLpIdx < laLpCnt; laLpIdx++) {
            SetUBZero(outLocal, arLenBA);
            InsertSync(HardEvent::V_MTE2);
            if (rLpCnt > 0) {
                copyParams.blockLen = (uint32_t)tilingPtr->raLpUnit * firstRLen * sizeof(T);
                gmOffset = (laLpIdx * tilingPtr->laOffset + curBlockIdx * tilingPtr->blockOffset -
                            excParams.excRFactor * tilingPtr->rOffset);
                gmOutOffset = gmOffset + excParams.excRFactor * tilingPtr->rOffset;
                (this->*func)(inGM, outGM, inLocal, outLocal, firstRLen, excParams.isExc, impl);
            }
            copyParams.blockLen = (uint32_t)tilingPtr->raLpUnit * rSize * sizeof(T);
            for (int64_t rLpIdx = 1; rLpIdx < rLpCnt; rLpIdx++) {
                gmOffset = (laLpIdx * tilingPtr->laOffset + curBlockIdx * tilingPtr->blockOffset +
                            (rLpIdx * rSize - excParams.excFactor) * tilingPtr->rOffset);
                gmOutOffset = gmOffset + excParams.excFactor * tilingPtr->rOffset;
                (this->*func)(inGM, outGM, inLocal, outLocal, rSize, false, impl);
            }
            if (rLeft > 0) {
                copyParams.blockLen = (uint32_t)tilingPtr->raLpUnit * rLeft * sizeof(T);
                gmOffset = (laLpIdx * tilingPtr->laOffset + curBlockIdx * tilingPtr->blockOffset +
                            (rLpCnt * rSize - excParams.excFactor) * tilingPtr->rOffset);
                gmOutOffset = gmOffset + excParams.excFactor * tilingPtr->rOffset;
                (this->*func)(inGM, outGM, inLocal, outLocal, rLeft, false, impl);
            }
        }
    } else {
        for (int64_t laLpIdx = 0; laLpIdx < laLpCnt; laLpIdx++) {
            SetUBZero(outLocal, arLenBA);
            InsertSync(HardEvent::V_MTE2);
            if (rLpCnt > 0) {
                copyParams.blockLen = (uint32_t)tilingPtr->raLpUnit * firstRLen * sizeof(T);
                gmOffset = (laLpIdx * tilingPtr->laOffset + curBlockIdx * tilingPtr->blockOffset +
                            (totalRSize - firstRLen) * tilingPtr->rOffset + excParams.excRFactor * tilingPtr->rOffset);
                gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
                (this->*func)(inGM, outGM, inLocal, outLocal, firstRLen, excParams.isExc, impl);
            }
            copyParams.blockLen = (uint32_t)tilingPtr->raLpUnit * rSize * sizeof(T);
            for (int64_t rLpIdx = rLpCnt - 1; rLpIdx > 0; rLpIdx--) {
                gmOffset = (laLpIdx * tilingPtr->laOffset + curBlockIdx * tilingPtr->blockOffset +
                            (rLpIdx * rSize + rLeft - rSize + excParams.excFactor) * tilingPtr->rOffset);
                gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
                (this->*func)(inGM, outGM, inLocal, outLocal, rSize, false, impl);
            }
            if (rLeft > 0) {
                copyParams.blockLen = (uint32_t)tilingPtr->raLpUnit * rLeft * sizeof(T);
                gmOffset = (laLpIdx * tilingPtr->laOffset + curBlockIdx * tilingPtr->blockOffset +
                            excParams.excFactor * tilingPtr->rOffset);
                gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
                (this->*func)(inGM, outGM, inLocal, outLocal, rLeft, false, impl);
            }
        }
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::CleanInputDirtyData(__local_mem__ T*& srcPtr, int32_t zeroBase, uint32_t leftSize)
{
    uint16_t lpCnt = Ops::Base::CeilDiv(leftSize, regElem);
    if constexpr (sizeof(T) == 1) {
        leftSize += leftSize;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<RangeType> sIdx;
        Reg::Arange(sIdx, zeroBase);
        Reg::RegTensor<T> zeros;
        Reg::Duplicate(zeros, (T)0);
        Reg::MaskReg mask;
        Reg::MaskReg idxMask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < lpCnt; i++) {
            mask = Reg::UpdateMask<T>(leftSize);
            Reg::DataCopyScatter(srcPtr, zeros, (Reg::RegTensor<IdxType>&)sIdx, mask);
            Reg::Adds(sIdx, sIdx, regElem, idxMask);
        }
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::TDRReverseProcess(
    __local_mem__ T*& srcPtr, __local_mem__ T*& dstPtr, uint16_t rLpNum, int32_t comRightA, int32_t splitRASize,
    int32_t curBegAddrBase, int32_t curBakBegAdd, uint16_t sndRLpCnt, int32_t phs1SrcBegIdx, int32_t phs2SrcBegIdx,
    int32_t phs1LpOffset, int32_t phs2LpOffset, int32_t phs2TLpOffset, int32_t bakBegAddr, int32_t phs1DstBegIdx,
    int32_t phs2DstBegIdx, uint32_t comLpMaskVal, uint32_t sndLpLeftMaskVal)
{
    int32_t rightALen = tilingPtr->raLpUnit;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::RegTensor<RangeType> gIdx;
        Reg::RegTensor<RangeType> sIdx;
        Reg::RegTensor<RangeType> tmp1;
        Reg::RegTensor<RangeType> tmp2;
        Reg::RegTensor<RangeType> gComIdx;
        Reg::MaskReg idxMask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();

        // get gather index: i - i//k*k + i//k*n*k
        Reg::Arange(gIdx, (RangeType)0);
        Reg::Duplicate(tmp1, (RangeType)rightALen);
        Reg::Div(tmp1, gIdx, tmp1, idxMask);
        Reg::Muls(tmp2, tmp1, splitRASize, idxMask);
        Reg::Muls(tmp1, tmp1, (RangeType)rightALen, idxMask); // tmp1 is i//k*k
        Reg::Sub(gIdx, gIdx, tmp1, idxMask);
        // the index result format is: (0, 1, 2, 0, 1, 2, 0, 1, 2, ...), concat last loop backup result
        Reg::Copy(gComIdx, gIdx); // backup gComIdx
        // the index result format is: (0, 1, 2, 6, 7, 8, ...)
        Reg::Add(gIdx, gIdx, tmp2, idxMask); // backup gIdx

        // get last loop result
        Reg::RegTensor<T> bakSrc;
        Reg::RegTensor<RangeType> gBakIdx;
        Reg::Adds(gBakIdx, gComIdx, bakBegAddr, idxMask);
        Reg::MaskReg comMask = Reg::UpdateMask<T>(comLpMaskVal);
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)bakSrc, dstPtr, (Reg::RegTensor<IdxType>&)gBakIdx, comMask);
        Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();

        // first operation, split R
        Reg::Duplicate(yDst, (T)0, comMask);
        Reg::Copy(sIdx, gIdx);
        Reg::Adds(gIdx, gIdx, phs1SrcBegIdx, idxMask);
        Reg::Adds(sIdx, sIdx, phs1DstBegIdx, idxMask);
        for (uint16_t rIdx = 0; rIdx < rLpNum; rIdx++) {
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, comMask);
            Reg::Add(yDst, yDst, xSrc, comMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, comMask);
            Reg::Adds(gIdx, gIdx, phs1LpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, phs1LpOffset, idxMask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        // second operation, get result
        Reg::Copy(xSrc, bakSrc);
        Reg::Arange(sIdx, (RangeType)0);
        Reg::Copy(gIdx, sIdx);
        Reg::Adds(gIdx, gIdx, phs2SrcBegIdx, idxMask);
        Reg::Adds(sIdx, sIdx, phs2DstBegIdx, idxMask);
        // for exclusive is true
        Reg::Adds(gComIdx, gComIdx, phs2DstBegIdx, idxMask);
        Reg::MaskReg sndLpLeftMask = Reg::UpdateMask<T>(sndLpLeftMaskVal);

        // first split RA of second phase
        for (uint16_t srIdx = 0; srIdx < sndRLpCnt; srIdx++) {
            Reg::Adds(gIdx, gIdx, phs2LpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, phs2LpOffset, idxMask);
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, comMask);
            Reg::Add(yDst, yDst, xSrc, comMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, comMask);
        }
        Reg::Adds(gIdx, gIdx, phs2TLpOffset, idxMask);
        Reg::Adds(sIdx, sIdx, phs2TLpOffset, idxMask);
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, sndLpLeftMask);
        Reg::Add(yDst, yDst, xSrc, sndLpLeftMask);
        Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, sndLpLeftMask);
        Reg::Adds(gComIdx, gComIdx, curBegAddrBase, idxMask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, dstPtr, (Reg::RegTensor<IdxType>&)gComIdx, comMask);

        for (uint16_t comIdx = 1; comIdx < (uint16_t)comRightA; comIdx++) {
            for (uint16_t srIdx = 0; srIdx < sndRLpCnt; srIdx++) {
                Reg::Adds(gIdx, gIdx, phs2LpOffset, idxMask);
                Reg::Adds(sIdx, sIdx, phs2LpOffset, idxMask);
                Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, comMask);
                Reg::Add(yDst, yDst, xSrc, comMask);
                Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, comMask);
            }
            Reg::Adds(gIdx, gIdx, phs2TLpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, phs2TLpOffset, idxMask);
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, sndLpLeftMask);
            Reg::Add(yDst, yDst, xSrc, sndLpLeftMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, sndLpLeftMask);
            Reg::Adds(gComIdx, gComIdx, curBakBegAdd, idxMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, dstPtr, (Reg::RegTensor<IdxType>&)gComIdx, comMask);
        }
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::TDRForwardProcess(
    __local_mem__ T*& srcPtr, __local_mem__ T*& dstPtr, uint16_t rLpNum, int32_t comRightA, int32_t splitRASize,
    int32_t curBegAddrBase, int32_t curBakBegAdd, uint16_t sndRLpCnt, int32_t phs1SrcBegIdx, int32_t phs2SrcBegIdx,
    int32_t phs1LpOffset, int32_t phs2LpOffset, int32_t phs2TLpOffset, int32_t bakBegAddr, int32_t phs1DstBegIdx,
    int32_t phs2DstBegIdx, uint32_t comLpMaskVal, uint32_t sndLpLeftMaskVal)
{
    int32_t rightALen = tilingPtr->raLpUnit;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::RegTensor<RangeType> gIdx;
        Reg::RegTensor<RangeType> sIdx;
        Reg::RegTensor<RangeType> tmp1;
        Reg::RegTensor<RangeType> tmp2;
        Reg::RegTensor<RangeType> gComIdx;
        Reg::MaskReg idxMask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();

        // get gather index: i - i//k*k + i//k*n*k
        Reg::Arange(gIdx, (RangeType)0);
        Reg::Duplicate(tmp1, (RangeType)rightALen);
        Reg::Div(tmp1, gIdx, tmp1, idxMask);
        Reg::Muls(tmp2, tmp1, splitRASize, idxMask);
        Reg::Muls(tmp1, tmp1, (RangeType)rightALen, idxMask); // tmp1 is i//k*k
        Reg::Sub(gIdx, gIdx, tmp1, idxMask);
        // the index result format is: (0, 1, 2, 0, 1, 2, 0, 1, 2, ...), concat last loop backup result
        Reg::Copy(gComIdx, gIdx); // backup gComIdx
        // the index result format is: (0, 1, 2, 6, 7, 8, ...)
        Reg::Add(gIdx, gIdx, tmp2, idxMask); // backup gIdx

        // get last loop result
        Reg::RegTensor<T> bakSrc;
        Reg::RegTensor<RangeType> gBakIdx;
        Reg::Adds(gBakIdx, gComIdx, bakBegAddr, idxMask);
        Reg::MaskReg comMask = Reg::UpdateMask<T>(comLpMaskVal);
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)bakSrc, dstPtr, (Reg::RegTensor<IdxType>&)gBakIdx, comMask);
        Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();

        // first operation, split R
        Reg::Duplicate(yDst, (T)0, comMask);
        Reg::Copy(sIdx, gIdx);
        Reg::Adds(gIdx, gIdx, phs1SrcBegIdx, idxMask);
        Reg::Adds(sIdx, sIdx, phs1DstBegIdx, idxMask);
        for (uint16_t rIdx = 0; rIdx < rLpNum; rIdx++) {
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, comMask);
            Reg::Add(yDst, yDst, xSrc, comMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, comMask);
            Reg::Adds(gIdx, gIdx, phs1LpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, phs1LpOffset, idxMask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

        // second operation, get result
        Reg::Copy(xSrc, bakSrc);
        Reg::Arange(sIdx, (RangeType)0);
        Reg::Copy(gIdx, sIdx);
        Reg::Adds(gIdx, gIdx, phs2SrcBegIdx, idxMask);
        Reg::Adds(sIdx, sIdx, phs2DstBegIdx, idxMask);
        // for exclusive is true
        Reg::Adds(gComIdx, gComIdx, phs2DstBegIdx, idxMask);
        Reg::MaskReg sndLpLeftMask = Reg::UpdateMask<T>(sndLpLeftMaskVal);

        // first split RA of second phase
        for (uint16_t srIdx = 0; srIdx < sndRLpCnt; srIdx++) {
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, comMask);
            Reg::Add(yDst, yDst, xSrc, comMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, comMask);
            Reg::Adds(gIdx, gIdx, phs2LpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, phs2LpOffset, idxMask);
        }
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, sndLpLeftMask);
        Reg::Add(yDst, yDst, xSrc, sndLpLeftMask);
        Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, sndLpLeftMask);
        Reg::Adds(gIdx, gIdx, phs2TLpOffset, idxMask);
        Reg::Adds(sIdx, sIdx, phs2TLpOffset, idxMask);
        Reg::Adds(gComIdx, gComIdx, curBegAddrBase, idxMask);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, dstPtr, (Reg::RegTensor<IdxType>&)gComIdx, comMask);

        for (uint16_t comIdx = 1; comIdx < (uint16_t)comRightA; comIdx++) {
            for (uint16_t srIdx = 0; srIdx < sndRLpCnt; srIdx++) {
                Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, comMask);
                Reg::Add(yDst, yDst, xSrc, comMask);
                Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, comMask);
                Reg::Adds(gIdx, gIdx, phs2LpOffset, idxMask);
                Reg::Adds(sIdx, sIdx, phs2LpOffset, idxMask);
            }
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gIdx, sndLpLeftMask);
            Reg::Add(yDst, yDst, xSrc, sndLpLeftMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, sndLpLeftMask);
            Reg::Adds(gIdx, gIdx, phs2TLpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, phs2TLpOffset, idxMask);
            Reg::Adds(gComIdx, gComIdx, curBakBegAdd, idxMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, dstPtr, (Reg::RegTensor<IdxType>&)gComIdx, comMask);
        }
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::CumGatherRightATDR(const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                        uint32_t rLen, bool isExc)
{
    /*  1. The input data layout in UB as follow, support copy data RA one time.
     *           |<--  rightALen -->|
     *           |------------------|
     *      rLen |------------------|
     *           |------------------|
     *  2. The position of result of last loop is at R.
     *  3. The rLen will be split into multiple parts.
     */

    __local_mem__ T* srcPtr = (__local_mem__ T*)inLocal.GetPhyAddr();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outLocal.GetPhyAddr();

    int32_t rightALen = tilingPtr->raLpUnit;
    uint32_t tmp = (regElem / rightALen > rLen) ? rLen : regElem / rightALen;
    uint16_t rLpNum = (uint16_t)Ops::Base::CeilDiv(rLen, tmp);
    int32_t comRightA = Ops::Base::CeilDiv(rLen, uint32_t(rLpNum));
    KCheckBGC(uint32_t(comRightA), uint32_t(rLpNum * rightALen));
    if (isBGC) {
        rLpNum = uint16_t((rLpNum * rightALen + blockElem + rightALen - 1) / rightALen);
        rLpNum = (uint16_t(rLen) > rLpNum) ? rLpNum : rLen;
        comRightA = Ops::Base::CeilDiv(rLen, uint32_t(rLpNum));
    }
    int32_t comLeftR = rLen % rLpNum;
    uint32_t comLpUnit = comRightA * rightALen;
    int32_t splitRASize = rLpNum * rightALen;
    int32_t curBegAddrBase = splitRASize - rightALen;
    int32_t curBakBegAdd = splitRASize;

    uint16_t sndRLpCnt = (uint16_t)(rLpNum / comRightA);
    int32_t sndRLpLeft = rLpNum % comRightA;
    if (sndRLpLeft == 0) {
        sndRLpCnt -= 1;
        sndRLpLeft = comRightA;
    }
    int32_t sndLpLeftSize = sndRLpLeft * rightALen;

    int32_t phs1SrcBegIdx = 0;
    int32_t phs2SrcBegIdx = 0;
    int32_t phs1LpOffset = rightALen;
    int32_t phs2LpOffset = comRightA * rightALen;
    int32_t phs2TLpOffset = sndLpLeftSize;
    int32_t bakBegAddr = (rSize - 1) * rightALen;
    if (tilingPtr->isReverse) {
        phs1SrcBegIdx = (rLpNum - 1) * rightALen;
        phs1LpOffset *= -1;
        phs2LpOffset *= -1;
        curBakBegAdd *= -1;
        phs2TLpOffset *= -1;
        bakBegAddr = 0;
        phs2SrcBegIdx = rLpNum * comRightA * rightALen;
        curBegAddrBase = -1 * rLpNum * rightALen;
    }
    int32_t phs1DstBegIdx = phs1SrcBegIdx;
    int32_t phs2DstBegIdx = phs2SrcBegIdx;
    if (isExc && !tilingPtr->isReverse) {
        phs1DstBegIdx = rightALen;
        phs2DstBegIdx = rightALen;
        phs2SrcBegIdx = rightALen;
    }

    // for update mask
    uint32_t comLpMaskVal = comLpUnit;
    uint32_t sndLpLeftMaskVal = sndLpLeftSize;
    if constexpr (sizeof(T) == 1) {
        comLpMaskVal += comLpMaskVal;
        sndLpLeftMaskVal += sndLpLeftMaskVal;
    }

    // clean dirty data
    if (tilingPtr->isReverse) {
        if (comLeftR > 0) {
            int32_t zeroBase = rLen * rightALen;
            uint32_t leftSize = (rLpNum - comLeftR) * rightALen;
            CleanInputDirtyData(srcPtr, zeroBase, leftSize);
            PipeBarrier<PIPE_V>();
        }
        TDRReverseProcess(srcPtr, dstPtr, rLpNum, comRightA, splitRASize, curBegAddrBase, curBakBegAdd, sndRLpCnt,
                          phs1SrcBegIdx, phs2SrcBegIdx, phs1LpOffset, phs2LpOffset, phs2TLpOffset, bakBegAddr,
                          phs1DstBegIdx, phs2DstBegIdx, comLpMaskVal, sndLpLeftMaskVal);
    } else {
        TDRForwardProcess(srcPtr, dstPtr, rLpNum, comRightA, splitRASize, curBegAddrBase, curBakBegAdd, sndRLpCnt,
                          phs1SrcBegIdx, phs2SrcBegIdx, phs1LpOffset, phs2LpOffset, phs2TLpOffset, bakBegAddr,
                          phs1DstBegIdx, phs2DstBegIdx, comLpMaskVal, sndLpLeftMaskVal);
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::CumGatherRightATDLeftA(const LocalTensor<T>& inLocal,
                                                            const LocalTensor<T>& outLocal, uint32_t leftALen,
                                                            uint32_t rLen, uint32_t laOffset, bool isExc)
{
    /*  1. The input data layout in UB as follow, support copy data RA or ARA one time.
     *           |<--  rightALen -->|
     *           |------------------|
     *      rLen |------------------| -----
     *           |------------------|     |
     *                    .               |
     *                    .              leftALen
     *                    .               |
     *           |------------------|     |
     *      rLen |------------------| -----
     *           |------------------|
     *  2. The position of result of last loop is at RAR+1 when isExc is true, otherwise the position is at R.
     *  3. The laOffset always be main loop size of RA.
     */

    __local_mem__ T* srcPtr = (__local_mem__ T*)inLocal.GetPhyAddr();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outLocal.GetPhyAddr();
    uint16_t rLpCnt = (uint16_t)rLen;
    uint32_t rightALen = tilingPtr->raLpUnit;
    uint32_t comRightACnt = (regElem / rightALen > leftALen) ? leftALen : regElem / rightALen;
    uint16_t aMainLpCnt = (uint16_t)(leftALen / comRightACnt);
    uint32_t leftATailCnt = leftALen % comRightACnt;
    if (leftATailCnt == 0) {
        aMainLpCnt -= 1;
        leftATailCnt = comRightACnt;
    }

    int32_t bakBegAddr = (rSize - 1) * tilingPtr->raLpUnit;
    int32_t srcBegAddr = 0;
    int32_t lpOffset = tilingPtr->raLpUnit;
    if (tilingPtr->isReverse) {
        bakBegAddr = 0;
        srcBegAddr = (rLen - 1) * tilingPtr->raLpUnit;
        lpOffset *= -1;
    }
    int32_t aLpIdxOffset = comRightACnt * laOffset;
    int32_t dstBegIdx = srcBegAddr;
    if (isExc && !tilingPtr->isReverse) {
        dstBegIdx += lpOffset;
    }

    // for mask
    uint32_t mainComSize = comRightACnt * rightALen;
    uint32_t tailComSize = leftATailCnt * rightALen;
    if constexpr (sizeof(T) == 1) {
        mainComSize += mainComSize;
        tailComSize += tailComSize;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::RegTensor<RangeType> gIdx;
        Reg::RegTensor<RangeType> gBakIdx;
        Reg::RegTensor<RangeType> tmp1;
        Reg::RegTensor<RangeType> tmp2;
        Reg::RegTensor<RangeType> sIdx;
        Reg::RegTensor<RangeType> sBakIdx;
        Reg::MaskReg idxMask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();

        /* get Gather index: i - i // rightA * rightA + i // rightA * laOffset
         * the index like (0, 1, 2, 30, 31, 32, 60, 61, 62)
         */
        Reg::Arange(gIdx, (RangeType)0);
        Reg::Duplicate(tmp1, (RangeType)rightALen);
        Reg::Div(tmp1, gIdx, tmp1, idxMask);
        Reg::Muls(tmp2, tmp1, (RangeType)laOffset, idxMask);
        Reg::Muls(tmp1, tmp1, (RangeType)rightALen, idxMask);
        Reg::Sub(gIdx, gIdx, tmp1, idxMask);
        Reg::Add(gIdx, gIdx, tmp2, idxMask); // backup gIdx
        Reg::Copy(gBakIdx, gIdx);
        Reg::Copy(sIdx, gIdx);
        Reg::Copy(sBakIdx, gIdx);
        Reg::Adds(gBakIdx, gBakIdx, bakBegAddr, idxMask);

        Reg::Adds(gIdx, gIdx, srcBegAddr, idxMask);
        Reg::Adds(sIdx, sIdx, dstBegIdx, idxMask);
        Reg::MaskReg mMask = Reg::UpdateMask<T>(mainComSize);
        Reg::MaskReg tMask = Reg::UpdateMask<T>(tailComSize);
        for (uint16_t aIdx = 0; aIdx < aMainLpCnt; aIdx++) {
            // get backup result of last loop
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gBakIdx, mMask);
            for (uint16_t rIdx = 0; rIdx < rLpCnt; rIdx++) {
                Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, mMask);
                Reg::Add(yDst, yDst, xSrc, mMask);
                Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, mMask);
                Reg::Adds(gIdx, gIdx, lpOffset, idxMask);
                Reg::Adds(sIdx, sIdx, lpOffset, idxMask);
            }
            Reg::Adds(gBakIdx, gBakIdx, aLpIdxOffset, idxMask);
            Reg::Adds(sBakIdx, sBakIdx, aLpIdxOffset, idxMask);
            Reg::Copy(gIdx, sBakIdx);
            Reg::Adds(sIdx, gIdx, dstBegIdx, idxMask);
            Reg::Adds(gIdx, gIdx, srcBegAddr, idxMask);
        }

        Reg::DataCopyGather((Reg::RegTensor<CastType>&)yDst, dstPtr, (Reg::RegTensor<IdxType>&)gBakIdx, tMask);
        for (uint16_t rIdx = 0; rIdx < rLpCnt; rIdx++) {
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, tMask);
            Reg::Add(yDst, yDst, xSrc, tMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)sIdx, tMask);
            Reg::Adds(gIdx, gIdx, lpOffset, idxMask);
            Reg::Adds(sIdx, sIdx, lpOffset, idxMask);
        }
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::TDLeftAInnerProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                         const LocalTensor<T>& inLocal, const LocalTensor<T>& outLocal,
                                                         int64_t laLen, int64_t rLen, bool isExc)
{
    if (tilingPtr->laOffset != tilingPtr->rLpUnit * tilingPtr->rOffset || isBGC || isExc) {
        if (rLen > 0) {
            CopyPara copyPara = {laLen, static_cast<uint32_t>(rLen * tilingPtr->rOffset), tilingPtr->laOffset, arLenBA,
                                 blockElem};
            CumCopyIn(inLocal, inGM[gmOffset], copyPara);
            InsertSync(HardEvent::MTE2_V);
            CumGatherRightATDLeftA(inLocal, outLocal, laLen, rLen, arLenBA, isExc);
            InsertSync(HardEvent::V_MTE2);
        }
        InsertSync(HardEvent::V_MTE3);
        if (isExc) {
            rLen += 1;
        }
        CopyPara copyPara = {laLen, static_cast<uint32_t>(rLen * tilingPtr->rOffset), tilingPtr->laOffset, arLenBA,
                             blockElem};
        CumCopyOut(outGM[gmOutOffset], outLocal, copyPara);
        InsertSync(HardEvent::MTE3_V);
    } else {
        uint32_t araLenBA = Ops::Base::CeilAlign(uint32_t(laLen * rLen * tilingPtr->rOffset), blockElem);
        if (rLen > 0) {
            CopyPara copyPara = {1, static_cast<uint32_t>(laLen * rLen * tilingPtr->rOffset), tilingPtr->laOffset,
                                 araLenBA, blockElem};
            CumCopyIn(inLocal, inGM[gmOffset], copyPara);
            InsertSync(HardEvent::MTE2_V);
            CumGatherRightATDLeftA(inLocal, outLocal, laLen, rLen, rLen * tilingPtr->rOffset, isExc);
            InsertSync(HardEvent::V_MTE2);
        }
        InsertSync(HardEvent::V_MTE3);
        CopyPara copyPara = {1, static_cast<uint32_t>(laLen * rLen * tilingPtr->rOffset), tilingPtr->laOffset, araLenBA,
                             blockElem};
        CumCopyOut(outGM[gmOutOffset], outLocal, copyPara);
        InsertSync(HardEvent::MTE3_V);
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::TDLeftAForwardProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                           const LocalTensor<T>& inLocal,
                                                           const LocalTensor<T>& outLocal, int64_t laLpIdx,
                                                           int64_t laSize)
{
    SetUBZero(outLocal, tilingPtr->laLpUnit * arLenBA);
    InsertSync(HardEvent::V_MTE2);
    if (rLpCnt > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset * tilingPtr->laLpUnit + curBlockIdx * tilingPtr->blockOffset -
                    excParams.excRFactor * tilingPtr->rOffset);
        gmOutOffset = gmOffset + excParams.excRFactor * tilingPtr->rOffset;
        TDLeftAInnerProcess(inGM, outGM, inLocal, outLocal, laSize, firstRLen, excParams.isExc);
    }
    for (int64_t rLpIdx = 1; rLpIdx < rLpCnt; rLpIdx++) {
        gmOffset = (laLpIdx * tilingPtr->laOffset * tilingPtr->laLpUnit + curBlockIdx * tilingPtr->blockOffset +
                    (rLpIdx * rSize - excParams.excFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset + excParams.excFactor * tilingPtr->rOffset;
        TDLeftAInnerProcess(inGM, outGM, inLocal, outLocal, laSize, rSize);
    }
    if (rLeft > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset * tilingPtr->laLpUnit + curBlockIdx * tilingPtr->blockOffset +
                    (rLpCnt * rSize - excParams.excFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset + excParams.excFactor * tilingPtr->rOffset;
        TDLeftAInnerProcess(inGM, outGM, inLocal, outLocal, laSize, rLeft);
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::TDLeftAReverseProcess(const GlobalTensor<T>& inGM, const GlobalTensor<T>& outGM,
                                                           const LocalTensor<T>& inLocal,
                                                           const LocalTensor<T>& outLocal, int64_t laLpIdx,
                                                           int64_t laSize)
{
    SetUBZero(outLocal, tilingPtr->laLpUnit * arLenBA);
    InsertSync(HardEvent::V_MTE2);
    if (rLpCnt > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset * tilingPtr->laLpUnit + curBlockIdx * tilingPtr->blockOffset +
                    (totalRSize - firstRLen) * tilingPtr->rOffset + excParams.excRFactor * tilingPtr->rOffset);
        gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
        TDLeftAInnerProcess(inGM, outGM, inLocal, outLocal, laSize, firstRLen, excParams.isExc);
    }
    for (int64_t rLpIdx = rLpCnt - 1; rLpIdx > 0; rLpIdx--) {
        gmOffset = (laLpIdx * tilingPtr->laOffset * tilingPtr->laLpUnit + curBlockIdx * tilingPtr->blockOffset +
                    (rLpIdx * rSize + rLeft - rSize + excParams.excFactor) * tilingPtr->rOffset);
        gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
        TDLeftAInnerProcess(inGM, outGM, inLocal, outLocal, laSize, rSize);
    }
    if (rLeft > 0) {
        gmOffset = (laLpIdx * tilingPtr->laOffset * tilingPtr->laLpUnit + curBlockIdx * tilingPtr->blockOffset +
                    excParams.excFactor * tilingPtr->rOffset);
        gmOutOffset = gmOffset - excParams.excFactor * tilingPtr->rOffset;
        TDLeftAInnerProcess(inGM, outGM, inLocal, outLocal, laSize, rLeft);
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::KCheckBGC(uint32_t comCnt, uint32_t arSize)
{
    constexpr uint32_t halfBGCnt = 8;
    constexpr uint32_t parallBytes = 512;
    uint32_t tmpComCnt = (halfBGCnt > comCnt) ? comCnt : halfBGCnt;
    uint32_t blockSize = blockElem * sizeof(T);
    if (arSize * sizeof(T) % blockSize == 0 && arSize * sizeof(T) * tmpComCnt % parallBytes == 0) {
        isBGC = true;
    } else {
        isBGC = false;
    }
}

template <typename T>
__aicore__ inline void GatherCum<T>::GatherRightATDLeftAProcess(const GlobalTensor<T>& inGM,
                                                                const GlobalTensor<T>& outGM,
                                                                const LocalTensor<T>& inLocal,
                                                                const LocalTensor<T>& outLocal)
{
    GetAxisLoopInfo();
    GetExclusiveInfo(tilingPtr, curBlockIdx, excParams, firstRLen, rLpCnt, rLeft);

    // to avoid bank group conflict
    uint32_t comCnt = ((regElem / int32_t(tilingPtr->raLpUnit) > uint32_t(tilingPtr->laLpUnit)) ?
                           uint32_t(tilingPtr->laLpUnit) :
                           regElem / int32_t(tilingPtr->raLpUnit));
    uint32_t tmpARSize = (tilingPtr->laOffset != tilingPtr->rLpUnit * tilingPtr->rOffset) ?
                             arLenBA :
                             uint32_t(tilingPtr->raLpUnit * rSize);
    KCheckBGC(comCnt, tmpARSize);
    if (isBGC) {
        arLenBA += blockElem;
    }

    if (!tilingPtr->isReverse) {
        for (int64_t laLpIdx = 0; laLpIdx < laLpCnt; laLpIdx++) {
            TDLeftAForwardProcess(inGM, outGM, inLocal, outLocal, laLpIdx, tilingPtr->laLpUnit);
        }
        if (laLeft > 0) {
            TDLeftAForwardProcess(inGM, outGM, inLocal, outLocal, laLpCnt, laLeft);
        }
    } else {
        for (int64_t laLpIdx = 0; laLpIdx < laLpCnt; laLpIdx++) {
            TDLeftAReverseProcess(inGM, outGM, inLocal, outLocal, laLpIdx, tilingPtr->laLpUnit);
        }
        if (laLeft > 0) {
            TDLeftAReverseProcess(inGM, outGM, inLocal, outLocal, laLpCnt, laLeft);
        }
    }
}

template <typename T>
class CumNoSplit {
public:
    __aicore__ inline CumNoSplit(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Cum4IntTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    const Cum4IntTilingData* tilingPtr;
    int32_t blockIdx = 0;
    TPipe* pipe;
    TQue<QuePosition::VECIN, queDepth> inQue;
    TQue<QuePosition::VECOUT, queDepth> outQue;
    GlobalTensor<T> inGM, outGM;
};

template <typename T>
__aicore__ inline void CumNoSplit<T>::Init(GM_ADDR x, GM_ADDR y, const Cum4IntTilingData* tilingDataPtr, TPipe* pipeIn)
{
    blockIdx = GetBlockIdx();
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe = pipeIn;
    tilingPtr = tilingDataPtr;
    pipe->InitBuffer(inQue, bufferNum, tilingPtr->tensorSize * sizeof(T));
    pipe->InitBuffer(outQue, bufferNum, tilingPtr->tensorSize * sizeof(T));
}

template <typename T>
__aicore__ inline void CumNoSplit<T>::Process()
{
    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();

    DataCopyTDRightACum<T> op(tilingPtr, blockIdx);
    op.CopyTDRightAProcess(inGM, outGM, inLocal, outLocal);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
class CumSplitAR {
public:
    __aicore__ inline CumSplitAR(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Cum4IntTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    const Cum4IntTilingData* tilingPtr;
    int32_t blockIdx = 0;
    TPipe* pipe;
    TQue<QuePosition::VECIN, queDepth> inQue;
    TQue<QuePosition::VECOUT, queDepth> outQue;
    GlobalTensor<T> inGM, outGM;
};

template <typename T>
__aicore__ inline void CumSplitAR<T>::Init(GM_ADDR x, GM_ADDR y, const Cum4IntTilingData* tilingDataPtr, TPipe* pipeIn)
{
    blockIdx = GetBlockIdx();
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe = pipeIn;
    tilingPtr = tilingDataPtr;
    pipe->InitBuffer(inQue, bufferNum, tilingPtr->tensorSize * sizeof(T));
    pipe->InitBuffer(outQue, bufferNum, tilingPtr->tensorSize * sizeof(T));
}

template <typename T>
__aicore__ inline void CumSplitAR<T>::Process()
{
    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();

    GatherCum<T> op(tilingPtr, pipe, blockIdx);
    op.Process(inGM, outGM, inLocal, outLocal);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
class CumWithGroup : public CumTypeBase<T> {
public:
    __aicore__ inline CumWithGroup(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Cum4IntTilingData* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void GetPrevCore();
    __aicore__ inline void GetPrevCoresResult(LocalTensor<T>& midUB, LocalTensor<T>& inUB, int64_t laIdx, int64_t raIdx,
                                              uint32_t raSize);
    __aicore__ inline void CrossCoreCum(const LocalTensor<T>& inUB, const LocalTensor<T>& midUB, LocalTensor<T>& outUB,
                                        uint32_t rLen, uint32_t raSize);
    __aicore__ inline void CrossCoreCum4BigRA(const LocalTensor<T>& inUB, const LocalTensor<T>& midUB,
                                              LocalTensor<T>& outUB, uint32_t rLen, uint32_t raSize);
    __aicore__ inline void AfterProcess(LocalTensor<T>& inUB, LocalTensor<T>& outUB);
    __aicore__ inline void InnerProcess4After(LocalTensor<T>& inUB, LocalTensor<T>& outUB, LocalTensor<T>& midUB,
                                              int64_t laIdx, int64_t raIdx, uint32_t raSize);

private:
    TBuf<QuePosition::VECCALC> midBuf;
    int64_t rLpCnt = 0;
    int64_t rLeft = 0;
    uint16_t prevCoreCnt = 0;
    int64_t gmOffset = 0;
    using IdxType = typename CumTypeBase<T>::IdxType;
    using RangeType = typename CumTypeBase<T>::RangeType;
    using VLType = typename CumTypeBase<T>::VLType;
    using CastType = typename CumTypeBase<T>::CastType;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    static constexpr uint32_t regElem = GetVecLen() / sizeof(VLType);
    static constexpr uint32_t realRegElem = GetVecLen() / sizeof(T);
    static constexpr uint32_t elemPerBlock = realRegElem / regBlock;
    const Cum4IntTilingData* tilingPtr;
    int32_t blockIdx = 0;
    TPipe* pipe;
    TQue<QuePosition::VECIN, queDepth> inQue;
    TQue<QuePosition::VECOUT, queDepth> outQue;
    GlobalTensor<T> inGM, outGM;
};

template <typename T>
__aicore__ inline void CumWithGroup<T>::Init(GM_ADDR x, GM_ADDR y, const Cum4IntTilingData* tilingDataPtr,
                                             TPipe* pipeIn)
{
    blockIdx = GetBlockIdx();
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe = pipeIn;
    tilingPtr = tilingDataPtr;
    pipe->InitBuffer(inQue, bufferNum, tilingPtr->tensorSize * sizeof(T));
    pipe->InitBuffer(midBuf, tilingPtr->raLpUnit * sizeof(T));
    pipe->InitBuffer(outQue, bufferNum, tilingPtr->tensorSize * sizeof(T));
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::Process()
{
    LocalTensor<T> inLocal = inQue.AllocTensor<T>();
    LocalTensor<T> outLocal = outQue.AllocTensor<T>();

    if (tilingPtr->raLpUnit >= regElem) {
        DataCopyTDRightACum<T> op(tilingPtr, blockIdx);
        op.CopyTDRightAProcess(inGM, outGM, inLocal, outLocal);
    } else {
        GatherCum<T> op(tilingPtr, pipe, blockIdx);
        op.Process(inGM, outGM, inLocal, outLocal);
    }
    AscendC::SyncAll();
    AfterProcess(inLocal, outLocal);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::GetPrevCore()
{
    if (tilingPtr->isReverse) {
        prevCoreCnt = (tilingPtr->usedCoreCnt - 1) - blockIdx;
    } else {
        prevCoreCnt = blockIdx;
    }
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::GetPrevCoresResult(LocalTensor<T>& midUB, LocalTensor<T>& inUB, int64_t laIdx,
                                                           int64_t raIdx, uint32_t raSize)
{
    if (prevCoreCnt < 1) {
        return;
    }

    uint32_t raBA = Ops::Base::CeilAlign(raSize, elemPerBlock);

    if (tilingPtr->isReverse) {
        gmOffset = (tilingPtr->ntcRLen * (blockIdx + 1) * tilingPtr->rOffset + laIdx * tilingPtr->laOffset +
                    raIdx * tilingPtr->raLpUnit);
        CopyPara copyPara = {prevCoreCnt, raSize, tilingPtr->ntcRLen * tilingPtr->rOffset, raBA, elemPerBlock};
        CumCopyIn(inUB, outGM[gmOffset], copyPara);
    } else {
        gmOffset = ((tilingPtr->ntcRLen - 1) * tilingPtr->rOffset + laIdx * tilingPtr->laOffset +
                    raIdx * tilingPtr->raLpUnit);
        CopyPara copyPara = {prevCoreCnt, raSize, tilingPtr->ntcRLen * tilingPtr->rOffset, raBA, elemPerBlock};
        CumCopyIn(inUB, outGM[gmOffset], copyPara);
    }
    InsertSync(HardEvent::MTE2_V);

    // rightA in UB is block align
    __local_mem__ T* midPtr = (__local_mem__ T*)midUB.GetPhyAddr();
    __local_mem__ T* inPtr = (__local_mem__ T*)inUB.GetPhyAddr();
    uint32_t rightALen = raSize;
    uint16_t raLpCnt_ = Ops::Base::CeilDiv(raSize, realRegElem);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> src;
        Reg::RegTensor<T> dst;
        Reg::Duplicate(dst, (T)0);
        Reg::MaskReg mask;
        for (uint16_t raIdx = 0; raIdx < raLpCnt_; raIdx++) {
            mask = Reg::UpdateMask<T>(rightALen);
            for (uint16_t idx = 0; idx < prevCoreCnt; idx++) {
                Reg::DataCopy(src, inPtr + raIdx * realRegElem + idx * raBA);
                Reg::Add(dst, dst, src, mask);
            }
            Reg::DataCopy<T>(midPtr + raIdx * realRegElem, dst, mask);
        }
    }
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::CrossCoreCum(const LocalTensor<T>& inUB, const LocalTensor<T>& midUB,
                                                     LocalTensor<T>& outUB, uint32_t rLen, uint32_t raSize)
{
    __local_mem__ T* srcPtr = (__local_mem__ T*)inUB.GetPhyAddr();
    __local_mem__ T* midPtr = (__local_mem__ T*)midUB.GetPhyAddr();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outUB.GetPhyAddr();
    // A must be smaller than VL
    uint32_t rightALen = raSize;
    uint32_t comA = regElem / rightALen;
    uint16_t rLpCnt = (uint16_t)(rLen / comA);
    uint32_t rLpLeft = rLen % comA;
    if (rLpLeft == 0) {
        rLpCnt -= 1;
        rLpLeft = comA;
    }
    int32_t mComA = comA * rightALen;

    // for mask
    uint32_t mMaskVal = comA * rightALen;
    uint32_t tMaskVal = rLpLeft * rightALen;
    if constexpr (sizeof(T) == 1) {
        mMaskVal += mMaskVal;
        tMaskVal += tMaskVal;
    }

    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::RegTensor<T> mSrc;
        Reg::RegTensor<RangeType> gIdx;
        Reg::RegTensor<RangeType> gBakIdx;
        Reg::MaskReg idxMask = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();

        Reg::Arange(gIdx, (RangeType)0);
        // gather index: i - i // k * k
        Reg::Duplicate(gBakIdx, (RangeType)rightALen);
        Reg::Div(gBakIdx, gIdx, gBakIdx, idxMask);
        Reg::Muls(gBakIdx, gBakIdx, (RangeType)rightALen, idxMask);
        Reg::Sub(gBakIdx, gIdx, gBakIdx, idxMask);

        Reg::MaskReg mMask = Reg::UpdateMask<T>(mMaskVal);
        // get last R result of previous cores
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)mSrc, midPtr, (Reg::RegTensor<IdxType>&)gBakIdx, mMask);
        for (uint16_t rIdx = 0; rIdx < rLpCnt; rIdx++) {
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, mMask);
            Reg::Add(yDst, xSrc, mSrc, mMask);
            Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)gIdx, mMask);
            Reg::Adds(gIdx, gIdx, mComA, idxMask);
        }
        Reg::MaskReg tMask = Reg::UpdateMask<T>(tMaskVal);
        Reg::DataCopyGather((Reg::RegTensor<CastType>&)xSrc, srcPtr, (Reg::RegTensor<IdxType>&)gIdx, tMask);
        Reg::Add(yDst, xSrc, mSrc, tMask);
        Reg::DataCopyScatter(dstPtr, yDst, (Reg::RegTensor<IdxType>&)gIdx, tMask);
    }
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::CrossCoreCum4BigRA(const LocalTensor<T>& inUB, const LocalTensor<T>& midUB,
                                                           LocalTensor<T>& outUB, uint32_t rLen, uint32_t raSize)
{
    __local_mem__ T* srcPtr = (__local_mem__ T*)inUB.GetPhyAddr();
    __local_mem__ T* midPtr = (__local_mem__ T*)midUB.GetPhyAddr();
    __local_mem__ T* dstPtr = (__local_mem__ T*)outUB.GetPhyAddr();

    uint32_t rightALen = raSize;
    uint16_t raLpCnt = (uint16_t)Ops::Base::CeilDiv(raSize, realRegElem);
    uint16_t rLpCnt = (uint16_t)rLen;
    uint32_t raBA = Ops::Base::CeilAlign(raSize, elemPerBlock);
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> xSrc;
        Reg::RegTensor<T> yDst;
        Reg::RegTensor<T> mSrc;
        Reg::MaskReg mask;

        // get last R result of previous cores
        Reg::DataCopy(mSrc, midPtr);
        uint32_t offset = 0;
        for (uint16_t raIdx = 0; raIdx < raLpCnt; raIdx++) {
            mask = Reg::UpdateMask<T>(rightALen);
            for (uint16_t rIdx = 0; rIdx < rLpCnt; rIdx++) {
                offset = raIdx * realRegElem + rIdx * raBA;
                Reg::DataCopy(xSrc, srcPtr + offset);
                Reg::Add(yDst, xSrc, mSrc, mask);
                Reg::DataCopy(dstPtr + offset, yDst, mask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::InnerProcess4After(LocalTensor<T>& inUB, LocalTensor<T>& outUB,
                                                           LocalTensor<T>& midUB, int64_t laIdx, int64_t raIdx,
                                                           uint32_t raSize)
{
    GetPrevCoresResult(midUB, inUB, laIdx, raIdx, raSize);
    uint32_t ubGap = Ops::Base::CeilAlign(raSize, elemPerBlock);
    bool isBigRA = (tilingPtr->raLpUnit >= regElem);

    AscendC::SyncAll();
    // 每次循环都要等所有核走到该处，所以不能直接退出
    if (prevCoreCnt > 0) {
        if (!isBigRA) {
            DataCopyExtParams copyParams{1, 0, 0, 0, 0};
            for (uint16_t rLpIdx = 0; rLpIdx < rLpCnt; rLpIdx++) {
                gmOffset = (laIdx * tilingPtr->laOffset + rLpIdx * tilingPtr->rLpUnit * tilingPtr->rOffset +
                            raIdx * tilingPtr->raLpUnit + blockIdx * tilingPtr->blockOffset);
                copyParams.blockLen = uint32_t(tilingPtr->rLpUnit * raSize * sizeof(T));
                DataCopyPad(inUB, outGM[gmOffset], copyParams, padParams);
                InsertSync(HardEvent::MTE2_V);
                CrossCoreCum(inUB, midUB, outUB, tilingPtr->rLpUnit, raSize);
                InsertSync(HardEvent::V_MTE2);
                InsertSync(HardEvent::V_MTE3);
                DataCopyPad(outGM[gmOffset], outUB, copyParams);
                InsertSync(HardEvent::MTE3_V);
            }
            if (rLeft > 0) {
                gmOffset = (laIdx * tilingPtr->laOffset + rLpCnt * tilingPtr->rLpUnit * tilingPtr->rOffset +
                            raIdx * tilingPtr->raLpUnit + blockIdx * tilingPtr->blockOffset);
                copyParams.blockLen = uint32_t(rLeft * raSize * sizeof(T));
                DataCopyPad(inUB, outGM[gmOffset], copyParams, padParams);
                InsertSync(HardEvent::MTE2_V);
                CrossCoreCum(inUB, midUB, outUB, rLeft, raSize);
                InsertSync(HardEvent::V_MTE2);
                InsertSync(HardEvent::V_MTE3);
                DataCopyPad(outGM[gmOffset], outUB, copyParams);
                InsertSync(HardEvent::MTE3_V);
            }
            InsertSync(HardEvent::MTE3_MTE2);
        } else {
            for (uint16_t rLpIdx = 0; rLpIdx < rLpCnt; rLpIdx++) {
                gmOffset = (laIdx * tilingPtr->laOffset + rLpIdx * tilingPtr->rLpUnit * tilingPtr->rOffset +
                            raIdx * tilingPtr->raLpUnit + blockIdx * tilingPtr->blockOffset);
                CopyPara copyInPara = {tilingPtr->rLpUnit, raSize, tilingPtr->rOffset, ubGap, elemPerBlock};
                CumCopyIn(inUB, outGM[gmOffset], copyInPara);
                InsertSync(HardEvent::MTE2_V);
                CrossCoreCum4BigRA(inUB, midUB, outUB, tilingPtr->rLpUnit, raSize);
                InsertSync(HardEvent::V_MTE2);
                InsertSync(HardEvent::V_MTE3);
                CopyPara copyOutPara = {tilingPtr->rLpUnit, raSize, tilingPtr->rOffset, ubGap, elemPerBlock};
                CumCopyOut(outGM[gmOffset], outUB, copyOutPara);
                InsertSync(HardEvent::MTE3_V);
            }
            if (rLeft > 0) {
                gmOffset = (laIdx * tilingPtr->laOffset + rLpCnt * tilingPtr->rLpUnit * tilingPtr->rOffset +
                            raIdx * tilingPtr->raLpUnit + blockIdx * tilingPtr->blockOffset);
                CopyPara copyInPara = {rLeft, raSize, tilingPtr->rOffset, ubGap, elemPerBlock};
                CumCopyIn(inUB, outGM[gmOffset], copyInPara);
                InsertSync(HardEvent::MTE2_V);
                CrossCoreCum4BigRA(inUB, midUB, outUB, rLeft, raSize);
                InsertSync(HardEvent::V_MTE2);
                InsertSync(HardEvent::V_MTE3);
                CopyPara copyOutPara = {rLeft, raSize, tilingPtr->rOffset, ubGap, elemPerBlock};
                CumCopyOut(outGM[gmOffset], outUB, copyOutPara);
                InsertSync(HardEvent::MTE3_V);
            }
            InsertSync(HardEvent::MTE3_MTE2);
        }
    }
}

template <typename T>
__aicore__ inline void CumWithGroup<T>::AfterProcess(LocalTensor<T>& inUB, LocalTensor<T>& outUB)
{
    LocalTensor<T> midUB = midBuf.Get<T>();

    if (tilingPtr->usedCoreCnt - 1 == blockIdx) {
        rLpCnt = tilingPtr->tcRLen / tilingPtr->rLpUnit;
        rLeft = tilingPtr->tcRLen % tilingPtr->rLpUnit;
    } else {
        rLpCnt = tilingPtr->ntcRLen / tilingPtr->rLpUnit;
        rLeft = tilingPtr->ntcRLen % tilingPtr->rLpUnit;
    }
    int64_t raLpCnt = tilingPtr->ntcRALen / tilingPtr->raLpUnit;
    int64_t raLeft = tilingPtr->ntcRALen % tilingPtr->raLpUnit;

    GetPrevCore();
    for (int64_t laIdx = 0; laIdx < tilingPtr->ntcLALen; laIdx++) {
        for (int64_t raIdx = 0; raIdx < raLpCnt; raIdx++) {
            InnerProcess4After(inUB, outUB, midUB, laIdx, raIdx, tilingPtr->raLpUnit);
        }
        if (raLeft > 0) {
            InnerProcess4After(inUB, outUB, midUB, laIdx, raLpCnt, raLeft);
        }
    }
}
} // namespace Cum

#endif // CUMSUM_4_INT_
