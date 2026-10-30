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
 * \file broadcast_to_with_tailAxis.h
 * \brief broadcst_to schedule
 */

#ifndef BROADCAST_TO_WITH_TAILAXIS_H_
#define BROADCAST_TO_WITH_TAILAXIS_H_

#include "broadcast_to_base.h"
#include "kernel_operator.h"

namespace BrcTo {
using namespace AscendC;

using AscendC::Reg::CreateMask;
using AscendC::Reg::DataCopy;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

constexpr int64_t DataCopyAlignUnit = 32;

template <typename T, typename U>
class BrcToWithTailAxis : public BrcToBase<U> {
public:
    __aicore__ inline BrcToWithTailAxis(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyDataIn();
    __aicore__ inline void ProcessBrc(int64_t aIdx, int64_t dataCnt);
    __aicore__ inline void BroadcastUb(LocalTensor<T> inTensor, int64_t inputOffset, int64_t elemIdx);
    __aicore__ inline void VFBrcTo(LocalTensor<T> outTensor, LocalTensor<T> inTensor, int64_t inputOffset,
                                   int64_t elemIdx, uint32_t brcCnt);
    __aicore__ inline void CopyDataOut(int64_t aIdx, int64_t elemIdx);

private:
    const U* tdPtr_;
    int64_t blockIdx = 0;
    AxesLpInfo lpInfo;
    int64_t aBaseIdx = 0;
    int64_t bBaseIdx = 0;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<T> inGM;
    GlobalTensor<T> outGM;
    int64_t gmInOffset;
    int64_t gmOutOffset;
    int64_t inBlockOffset = 0;
    int64_t outBlockOffset = 0;
    AscendC::DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe_ = pipeIn;
    tdPtr_ = tilingDataPtr;
    pipe_->InitBuffer(inQueue_, tdPtr_->bufferCnt, tdPtr_->tensorSize * sizeof(T));
    pipe_->InitBuffer(outQueue_, tdPtr_->bufferCnt, tdPtr_->tensorSize * sizeof(T));
    blockIdx = GetBlockIdx() % tdPtr_->usedCoreCnt;
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    this->CalcInBlockOffset(tdPtr_, blockIdx, inBlockOffset);
    this->CalcOutBlockOffset(tdPtr_, blockIdx, outBlockOffset);

    int64_t aLpCnt = lpInfo.aLpCnt / tdPtr_->aLpUnit;
    int64_t aLpLeft = lpInfo.aLpCnt % tdPtr_->aLpUnit;

    for (int64_t aIdx = 0; aIdx < aLpCnt; aIdx++) {
        gmInOffset = (this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx * tdPtr_->aLpUnit) + inBlockOffset);
        copyInParams_.blockLen = tdPtr_->aLpUnit * sizeof(T);
        CopyDataIn();
        ProcessBrc(aIdx, tdPtr_->aLpUnit);
    }
    if (aLpLeft > 0) {
        gmInOffset = (this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aLpCnt * tdPtr_->aLpUnit) + inBlockOffset);
        copyInParams_.blockLen = aLpLeft * sizeof(T);
        CopyDataIn();
        ProcessBrc(aLpCnt, aLpLeft);
    }
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::CopyDataIn()
{
    LocalTensor<T> inTensor = inQueue_.AllocTensor<T>();
    AscendC::DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    DataCopyPad(inTensor, inGM[gmInOffset], copyInParams_, copyInPadParams);
    inQueue_.EnQue(inTensor);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::ProcessBrc(int64_t aIdx, int64_t dataCnt)
{
    LocalTensor<T> inTensor = inQueue_.DeQue<T>();

    int64_t copyUnit = DataCopyAlignUnit / sizeof(T);
    int64_t copyLpCnt = dataCnt / copyUnit;
    int64_t copyLpLeft = dataCnt % copyUnit;

    int64_t inputOffset = 0;

    for (int64_t copyIdx = 0; copyIdx < copyLpCnt; copyIdx++) {
        inputOffset = copyIdx * copyUnit;
        for (int64_t elemIdx = 0; elemIdx < copyUnit; elemIdx++) {
            BroadcastUb(inTensor, inputOffset, elemIdx);
            CopyDataOut(aIdx, inputOffset + elemIdx);
        }
    }
    if (copyLpLeft > 0) {
        inputOffset = copyLpCnt * copyUnit;
        for (int64_t elemIdx = 0; elemIdx < copyLpLeft; elemIdx++) {
            BroadcastUb(inTensor, inputOffset, elemIdx);
            CopyDataOut(aIdx, inputOffset + elemIdx);
        }
    }
    inQueue_.FreeTensor(inTensor);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::BroadcastUb(LocalTensor<T> inTensor, int64_t inputOffset,
                                                            int64_t elemIdx)
{
    LocalTensor<T> outTensor = outQueue_.AllocTensor<T>();

    VFBrcTo(outTensor, inTensor, inputOffset, elemIdx, static_cast<uint32_t>(tdPtr_->tensorSize));

    outQueue_.EnQue(outTensor);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::CopyDataOut(int64_t aIdx, int64_t elemIdx)
{
    LocalTensor<T> tensor = outQueue_.DeQue<T>();
    AscendC::DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx * tdPtr_->aLpUnit + elemIdx);
    int64_t bOffset = 0;

    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        copyOutParams_.blockLen = tdPtr_->uLpUnit * tdPtr_->uOutOffset * sizeof(T);
        for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
            gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            DataCopyPad(outGM[gmOutOffset + bOffset], tensor, copyOutParams_);
        }
        if (lpInfo.uLeft > 0) {
            copyOutParams_.blockLen = lpInfo.uLeft * tdPtr_->uOutOffset * sizeof(T);
            gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
            DataCopyPad(outGM[gmOutOffset + bOffset], tensor, copyOutParams_);
        }
    }
    outQueue_.FreeTensor(tensor);
}

template <typename T, typename U>
__aicore__ inline void BrcToWithTailAxis<T, U>::VFBrcTo(LocalTensor<T> outTensor, LocalTensor<T> inTensor,
                                                        int64_t inputOffset, int64_t elemIdx, uint32_t brcCnt)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)outTensor.GetPhyAddr();

    uint32_t VL_CNT = Ops::Base::GetVRegSize() / sizeof(T);
    uint16_t brcLoopCnt = Ops::Base::CeilDiv(brcCnt, VL_CNT);
    int64_t outputOffset = 0;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> tmpIn;
        AscendC::Reg::RegTensor<T> tmpOut;
        MaskReg pregAll = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregAllB8 = CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg pregAllFB8 = CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALLF>();

        MaskReg pregTmp;
        MaskReg pregGather;
        MaskReg pregLoopB;

        DataCopy(tmpIn, inputAddr + inputOffset);
        uint32_t sregTmp = uint32_t(elemIdx * sizeof(T));
        pregTmp = UpdateMask<uint8_t>(sregTmp);
        AscendC::Reg::MaskSel(pregGather, pregAllFB8, pregAllB8, pregTmp);
        GatherMask((Reg::RegTensor<uint8_t>&)tmpOut, (Reg::RegTensor<uint8_t>&)tmpIn, pregGather);
        Duplicate(tmpOut, tmpOut, pregAll);

        uint32_t sregB = brcCnt;
        for (uint16_t vIdx = 0; vIdx < brcLoopCnt; vIdx++) {
            pregLoopB = AscendC::Reg::UpdateMask<T>(sregB);
            AscendC::Reg::DataCopy(outputAddr + outputOffset, tmpOut, pregLoopB);
            outputOffset += VL_CNT;
        }
    }
}

} // namespace BrcTo

#endif
