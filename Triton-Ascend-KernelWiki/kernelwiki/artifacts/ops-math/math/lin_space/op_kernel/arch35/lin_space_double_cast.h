/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_double_cast.h
 * \brief
 */

#ifndef LIN_SPACE_DOUBLE_CAST_H
#define LIN_SPACE_DOUBLE_CAST_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/load_store_utils.h"

namespace LinSpace {
using namespace AscendC;
constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t DB_BUFFER = 2;

template <typename C, typename S>
class LinSpaceDoubleCast {
public:
    __aicore__ inline LinSpaceDoubleCast(){};
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output, GM_ADDR workspace,
                                const LinSpaceRegbaseTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process(const LinSpaceRegbaseTilingData* tilingData);

private:
    __aicore__ inline void Compute(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void ComputeVf(const LocalTensor<S>& dstUb, const LocalTensor<C>& srcUb, int64_t loopIdx,
                                     uint32_t count, float offset);
    __aicore__ inline void ComputeBackward(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyOutBackward(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyOutBackwardTail(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void GenSequence(int64_t dataCount);
    __aicore__ inline void GenLinSpace(int64_t loopCnt, int64_t tailOfCurCore, bool isForward);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outDataQueue_;
    TBuf<QuePosition::VECCALC> sequenceBuf_;
    GlobalTensor<S> outputGm_;
    int64_t num_{0};
    C start_{0};
    C stop_{0};
    C step_{0};
    C firstValue_{0};
    int64_t curNumOfCore_{0};
    int64_t curPerOfCore_{0};
    int64_t curTailOfCore_{0};
    int64_t loopCount_{0};
    int64_t coreOffsetForward_{0};
    int64_t coreOffsetBackward_{0};
    int64_t coreOffsetBackwardConvert_{0};
    int64_t forwardNumOfHalfWayCore_{0};
    int64_t loopOfForward_{0};
    int64_t tailOfForward_{0};
    int64_t loopOfBackward_{0};
    int64_t tailOfBackward_{0};
};

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::Init(GM_ADDR start, GM_ADDR stop, GM_ADDR num, GM_ADDR output,
                                                      GM_ADDR workspace, const LinSpaceRegbaseTilingData* tilingData,
                                                      TPipe* pipeIn)
{
    if (tilingData->num < 1) {
        return;
    }
    if (GetBlockIdx() == tilingData->halfWayCoreIdx) {
        forwardNumOfHalfWayCore_ = tilingData->forwardNumOfHalfWayCore;
        loopOfForward_ = tilingData->loopOfForward;
        tailOfForward_ = tilingData->tailOfForward;
        loopOfBackward_ = tilingData->loopOfBackward;
        tailOfBackward_ = tilingData->tailOfBackward;
    }
    if (GetBlockIdx() + 1 == tilingData->usedCoreNum) {
        curNumOfCore_ = tilingData->numOfTailCore;
        curPerOfCore_ = tilingData->perOfTailCore;
        curTailOfCore_ = tilingData->tailOfTailCore;
        loopCount_ = tilingData->loopOfTailCore;
    } else {
        curNumOfCore_ = tilingData->numOfPerCore;
        curPerOfCore_ = tilingData->perOfPerCore;
        curTailOfCore_ = tilingData->tailOfPerCore;
        loopCount_ = tilingData->loopOfPerCore;
    }
    num_ = tilingData->num;
    start_ = static_cast<C>(tilingData->start);
    stop_ = static_cast<C>(tilingData->stop);
    step_ = static_cast<C>(tilingData->step);
    coreOffsetForward_ = GetBlockIdx() * tilingData->numOfPerCore;
    coreOffsetBackward_ = num_ - coreOffsetForward_ - curNumOfCore_;
    // SetBuffer
    outputGm_.SetGlobalBuffer((__gm__ S*)output + coreOffsetForward_, curNumOfCore_);
    // InitBuffer
    pipe_ = pipeIn;
    pipe_->InitBuffer(outDataQueue_, DB_BUFFER, Ops::Base::CeilAlign(curPerOfCore_, ONCE_ALGN_NUM_INT32) * sizeof(S));
    pipe_->InitBuffer(sequenceBuf_, Ops::Base::CeilAlign(curPerOfCore_, ONCE_ALGN_NUM_INT32) * sizeof(C));
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::GenSequence(int64_t dataCount)
{
    LocalTensor<C> sequenceUb = sequenceBuf_.Get<C>();
    const C firstValue = static_cast<C>(0);
    uint32_t vciCount = static_cast<uint32_t>(dataCount);
    CreateVecIndex(sequenceUb, firstValue, vciCount);
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::ComputeVf(const LocalTensor<S>& dstUb, const LocalTensor<C>& srcUb,
                                                           int64_t loopIdx, uint32_t count, float offset)
{
    uint32_t dtypeSize = sizeof(C);
    uint32_t VL = Ops::Base::GetVRegSize() / dtypeSize;
    uint16_t loopNum = (count + VL - 1) / VL;
    uint32_t vlSize = VL;
    float stepValue = step_;
    float firstValue = firstValue_;
    __local_mem__ C* srcAddr = (__local_mem__ C*)srcUb.GetPhyAddr();
    __local_mem__ S* dstAddr = (__local_mem__ S*)dstUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<C, AscendC::Reg::RegTraitNumOne> vregInput;
        AscendC::Reg::RegTensor<C, AscendC::Reg::RegTraitNumOne> vregAdd;
        AscendC::Reg::RegTensor<C, AscendC::Reg::RegTraitNumOne> vregFloat;
        AscendC::Reg::MaskReg mask;

        for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
            mask = AscendC::Reg::UpdateMask<C, AscendC::Reg::RegTraitNumOne>(count);
            AscendC::Reg::DataCopy(vregInput, (__ubuf__ C*)(srcAddr + loopIdx * vlSize));
            AscendC::Reg::Adds(vregAdd, vregInput, offset, mask);
            AscendC::Reg::Duplicate(vregFloat, firstValue, mask);
            AscendC::Reg::Axpy(vregFloat, vregAdd, stepValue, mask);
            ops::StoreOneTensorForDtypeT<S>(dstAddr, vregFloat, mask, loopIdx * vlSize);
        }
    }
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::Compute(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<S> outDataUbForward = outDataQueue_.AllocTensor<S>();
    LocalTensor<C> sequenceUbForward = sequenceBuf_.Get<C>();
    // 等差数列的基本公式，a_n = a_1 + (n - 1) * d, 实现乘加操作
    float curOffsetForward = static_cast<C>(coreOffsetForward_ + loopIdx * curPerOfCore_);
    uint32_t count = static_cast<uint32_t>(dataCount);
    ComputeVf(outDataUbForward, sequenceUbForward, loopIdx, count, curOffsetForward);
    outDataQueue_.EnQue<S>(outDataUbForward);
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::ComputeBackward(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<S> outDataUbBackward = outDataQueue_.AllocTensor<S>();
    LocalTensor<C> sequenceUbBackward = sequenceBuf_.Get<C>();
    // 等差数列的基本公式，a_n = a_e - (s - n) * d
    float curOffsetBackward = static_cast<C>(1 - coreOffsetBackward_ - loopIdx * curPerOfCore_ - dataCount);
    uint32_t count = static_cast<uint32_t>(dataCount);
    ComputeVf(outDataUbBackward, sequenceUbBackward, loopIdx, count, curOffsetBackward);
    outDataQueue_.EnQue<S>(outDataUbBackward);
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::CopyOut(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<S> outDataUbForward = outDataQueue_.DeQue<S>();
    DataCopyExtParams copyParamsYOut{static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(S)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[loopIdx * curPerOfCore_], outDataUbForward, copyParamsYOut);
    outDataQueue_.FreeTensor(outDataUbForward);
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::CopyOutBackward(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<S> outDataUbBackward = outDataQueue_.DeQue<S>();
    DataCopyExtParams copyParamsYOut{static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(S)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[curNumOfCore_ - ((loopIdx + 1) * curPerOfCore_)], outDataUbBackward, copyParamsYOut);
    outDataQueue_.FreeTensor(outDataUbBackward);
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::CopyOutBackwardTail(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<S> outDataUbBackwardTail = outDataQueue_.DeQue<S>();
    DataCopyExtParams copyParamsYOut{static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(S)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[coreOffsetBackwardConvert_], outDataUbBackwardTail, copyParamsYOut);
    outDataQueue_.FreeTensor(outDataUbBackwardTail);
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::GenLinSpace(int64_t loopCnt, int64_t tailOfCurCore, bool isForward)
{
    if (isForward) {
        firstValue_ = start_;
        for (int64_t n = 0; n < loopCnt - 1; n++) {
            Compute(n, curPerOfCore_);
            CopyOut(n, curPerOfCore_);
        }
        {
            Compute(loopCnt - 1, tailOfCurCore);
            CopyOut(loopCnt - 1, tailOfCurCore);
        }
    } else {
        firstValue_ = stop_;
        for (int64_t n = 0; n < loopCnt - 1; n++) {
            ComputeBackward(n, curPerOfCore_);
            CopyOutBackward(n, curPerOfCore_);
        }
        {
            ComputeBackward(loopCnt - 1, tailOfCurCore);
            CopyOutBackwardTail(loopCnt - 1, tailOfCurCore);
        }
    }
}

template <typename C, typename S>
__aicore__ inline void LinSpaceDoubleCast<C, S>::Process(const LinSpaceRegbaseTilingData* tilingData)
{
    if (GetBlockIdx() >= tilingData->usedCoreNum || tilingData->num < 1) {
        return;
    }
    GenSequence(curPerOfCore_);
    if (GetBlockIdx() == tilingData->halfWayCoreIdx) {
        coreOffsetBackwardConvert_ = forwardNumOfHalfWayCore_;
        GenLinSpace(loopOfForward_, tailOfForward_, true);
        GenLinSpace(loopOfBackward_, tailOfBackward_, false);
    } else if (GetBlockIdx() < tilingData->halfWayCoreIdx) {
        GenLinSpace(loopCount_, curTailOfCore_, true);
    } else {
        GenLinSpace(loopCount_, curTailOfCore_, false);
    }
}

} // namespace LinSpace
#endif // LIN_SPACE_DOUBLE_CAST_H
