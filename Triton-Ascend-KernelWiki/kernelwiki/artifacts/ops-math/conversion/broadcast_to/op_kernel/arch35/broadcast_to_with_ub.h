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
 * \file broadcast_to_with_ub.h
 * \brief broadcst_to schedule
 */

#ifndef BROADCAST_TO_WITH_UB_H_
#define BROADCAST_TO_WITH_UB_H_

#include <type_traits>

#include "broadcast_to_base.h"
#include "kernel_operator.h"

namespace BrcTo {
using namespace AscendC;

using AscendC::Reg::CreateMask;
using AscendC::Reg::DataCopy;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

constexpr uint32_t MAX_INNER_DIM = 5;
constexpr uint32_t DIM_TWO = 2;
constexpr uint32_t DIM_THREE = 3;
constexpr uint32_t DIM_FOUR = 4;
constexpr uint32_t DATA_ALIGN_SIZE = 32;

template <typename T, typename U, bool isLastDimSmall = false>
class BroadcastToUb : public BrcToBase<U> {
public:
    __aicore__ inline BroadcastToUb(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyDataIn(LocalTensor<T>& tensor);
    __aicore__ inline void CopyDataOut(LocalTensor<T>& tensor);
    __aicore__ inline void CalcInnerShape();
    __aicore__ inline void SetOutputParams(uint32_t uLen);
    __aicore__ inline void SetInputParams(uint32_t uLen);
    __aicore__ inline void VFBroadcastOneElemLB(__local_mem__ T* inputAddr, __local_mem__ T* outputAddr);
    __aicore__ inline void VFBroadcastOneElemLBB64(__local_mem__ T* inputAddr, __local_mem__ T* outputAddr);
    __aicore__ inline void VFBroadcastOneElemOB(__local_mem__ T* outputAddr);
    __aicore__ inline void VFInnerBroadcastToB(LocalTensor<T>& outputTensor, LocalTensor<T>& inputTensor);
    __aicore__ inline void VFInnerBroadcastToA(LocalTensor<T>& outputTensor, LocalTensor<T>& inputTensor);
    __aicore__ inline void VFInnerBrcLastDimLEBlock(LocalTensor<T>& outputTensor, LocalTensor<T>& inputTensor);
    __aicore__ inline void GenGatherIdx(Reg::RegTensor<int32_t>& gatherIdx, int32_t axis4BA);
    __aicore__ inline void VFInnerBrcLastDimGTBlock(LocalTensor<T>& outputTensor, LocalTensor<T>& inputTensor);
    __aicore__ inline void BroadcastUb(LocalTensor<T>& outputTensor);
    __aicore__ inline void BrcUNotB(int64_t aIdx);
    __aicore__ inline void BrcUIsB(int64_t aIdx);

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
    AscendC::DataCopyPadExtParams<T> copyInPadParams_{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    uint32_t innerShape[MAX_INNER_DIM] = {1, 1, 1, 1, 1};
    uint32_t innerAxis0 = 1;
    uint32_t innerAxis1 = 1;
    uint32_t innerAxis2 = 1;
    uint32_t innerAxis3 = 1;
    uint32_t innerAxis4 = 1;
    uint32_t dataAlignCnt = 0;
    uint32_t VL_CNT = 0;
    using RT = std::conditional_t<sizeof(T) != sizeof(uint64_t), T, uint32_t>;
};

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::Init(GM_ADDR x, GM_ADDR y, const U* tilingDataPtr,
                                                                 TPipe* pipeIn)
{
    inGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe_ = pipeIn;
    tdPtr_ = tilingDataPtr;
    pipe_->InitBuffer(inQueue_, tdPtr_->bufferCnt, tdPtr_->tensorSize * sizeof(T));
    pipe_->InitBuffer(outQueue_, tdPtr_->bufferCnt, tdPtr_->tensorSize * sizeof(T));
    dataAlignCnt = DATA_ALIGN_SIZE / sizeof(T);
    VL_CNT = Ops::Base::GetVRegSize() / sizeof(T);
    blockIdx = GetBlockIdx() % tdPtr_->usedCoreCnt;
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::CopyDataIn(LocalTensor<T>& tensor)
{
    DataCopyPad(tensor, inGM[gmInOffset], copyInParams_, copyInPadParams_);
    inQueue_.EnQue(tensor);
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::CopyDataOut(LocalTensor<T>& tensor)
{
    int64_t bOffset = 0;
    for (int64_t bIdx = lpInfo.bLpBegIdx; bIdx < lpInfo.bLpEndIdx; bIdx++) {
        bOffset = this->CalcBAxesOffset(tdPtr_, bBaseIdx + bIdx);
        DataCopyPad(outGM[gmOutOffset + bOffset], tensor, copyOutParams_);
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::CalcInnerShape()
{
    for (uint8_t i = 0; i < tdPtr_->uAxisCnt; i++) {
        innerShape[MAX_INNER_DIM - tdPtr_->uAxisCnt + i] = tdPtr_->xSize[i];
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::SetOutputParams(uint32_t uLen)
{
    if (innerShape[DIM_FOUR] % dataAlignCnt != 0) {
        copyOutParams_.blockCount = uLen * uint32_t(tdPtr_->uOutOffset) / innerShape[DIM_FOUR];
        copyOutParams_.blockLen = innerShape[DIM_FOUR] * sizeof(T);
        return;
    }
    copyOutParams_.blockLen = uLen * tdPtr_->uOutOffset * sizeof(T);
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::SetInputParams(uint32_t uLen)
{
    if (tdPtr_->isLastDimB == 0 && innerShape[DIM_FOUR] % dataAlignCnt != 0U) {
        copyInParams_.blockCount = uLen * uint32_t(tdPtr_->uInOffset) / innerShape[DIM_FOUR];
        copyInParams_.blockLen = innerShape[DIM_FOUR] * sizeof(T);
        return;
    }
    copyInParams_.blockLen = uLen * tdPtr_->uInOffset * sizeof(T);
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFBroadcastOneElemLB(__local_mem__ T* inputAddr,
                                                                                 __local_mem__ T* outputAddr)
{
    uint32_t axis3OutOffset = Ops::Base::CeilAlign(innerAxis4, dataAlignCnt);
    uint32_t axis1OutOffset = innerAxis2 * innerAxis3 * axis3OutOffset;
    uint16_t axis4LpCnt = Ops::Base::CeilDiv(innerAxis4, VL_CNT);
    uint16_t axis4Offset = VL_CNT;
    uint32_t maskValue = axis3OutOffset;
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> tmpIn;
        AscendC::Reg::RegTensor<T> tmpOut;
        MaskReg mask;
        for (uint16_t axis4LpIdx = 0; axis4LpIdx < axis4LpCnt; axis4LpIdx++) {
            mask = UpdateMask<T>(maskValue);
            for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(innerAxis1); axis1Idx++) {
                for (uint16_t axis3Idx = 0; axis3Idx < static_cast<uint16_t>(innerAxis3); axis3Idx++) {
                    auto aregI = Reg::CreateAddrReg<T>(axis1Idx, innerAxis3, axis3Idx, 1);
                    if constexpr (sizeof(T) == sizeof(uint8_t)) {
                        DataCopy<T, Reg::LoadDist::DIST_BRC_B8>(tmpIn, inputAddr, aregI);
                    } else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                        DataCopy<T, Reg::LoadDist::DIST_BRC_B16>(tmpIn, inputAddr, aregI);
                    } else {
                        DataCopy<T, Reg::LoadDist::DIST_BRC_B32>(tmpIn, inputAddr, aregI);
                    }
                    auto aregO = Reg::CreateAddrReg<T>(axis4LpIdx, axis4Offset, axis1Idx, axis1OutOffset, axis3Idx,
                                                       axis3OutOffset);
                    DataCopy(outputAddr, tmpIn, aregO, mask);
                }
            }
        }
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFBroadcastOneElemLBB64(__local_mem__ T* inputAddr,
                                                                                    __local_mem__ T* outputAddr)
{
    uint32_t axis1InOffset = innerAxis3 * nTwo;
    uint32_t axis3OutOffset = Ops::Base::CeilAlign(innerAxis4, dataAlignCnt) * nTwo;
    uint32_t axis1OutOffset = innerAxis2 * innerAxis3 * axis3OutOffset;
    uint16_t axis4LpCnt = Ops::Base::CeilDiv(innerAxis4, VL_CNT);
    uint16_t axis4Offset = VL_CNT * nTwo;
    uint32_t maskValue = axis3OutOffset;
    auto reInAddr = reinterpret_cast<__local_mem__ RT*>(inputAddr);
    auto reOutAddr = reinterpret_cast<__local_mem__ RT*>(outputAddr);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<RT> tmpIn;
        AscendC::Reg::RegTensor<RT> tmpIn1;
        AscendC::Reg::RegTensor<RT> tmpOut;
        AscendC::Reg::RegTensor<RT> tmpOut1;
        MaskReg mask;
        for (uint16_t axis4LpIdx = 0; axis4LpIdx < axis4LpCnt; axis4LpIdx++) {
            mask = UpdateMask<RT>(maskValue);
            for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(innerAxis1); axis1Idx++) {
                for (uint16_t axis3Idx = 0; axis3Idx < static_cast<uint16_t>(innerAxis3); axis3Idx++) {
                    auto regI = Reg::CreateAddrReg<RT>(axis1Idx, axis1InOffset, axis3Idx, nTwo);
                    DataCopy<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn, reInAddr, regI);
                    DataCopy<RT, Reg::LoadDist::DIST_BRC_B32>(tmpIn1, reInAddr + 1, regI);
                    Reg::Interleave(tmpOut, tmpOut1, tmpIn, tmpIn1);
                    auto aregO = Reg::CreateAddrReg<RT>(axis4LpIdx, axis4Offset, axis1Idx, axis1OutOffset, axis3Idx,
                                                        axis3OutOffset);
                    DataCopy(reOutAddr, tmpOut, aregO, mask);
                }
            }
        }
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFBroadcastOneElemOB(__local_mem__ T* outputAddr)
{
    uint32_t axis4BA = Ops::Base::CeilAlign(innerAxis4, dataAlignCnt);
    uint32_t axis2Offset = innerAxis3 * axis4BA;
    uint32_t iAxis34Size = innerAxis3 * axis4BA;
    uint16_t axis34LpCnt = Ops::Base::CeilDiv(iAxis34Size, VL_CNT);
    uint32_t axis34Offset = VL_CNT;
    if constexpr (sizeof(T) != sizeof(RT)) {
        axis34Offset *= nTwo;
        axis2Offset *= nTwo;
        iAxis34Size *= nTwo;
    }
    uint32_t axis1OutOffset = innerAxis2 * axis2Offset;
    auto reOutAddr = reinterpret_cast<__local_mem__ RT*>(outputAddr);
    // ABA
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<RT> tmpIn;
        for (uint16_t axis34LpIdx = 0; axis34LpIdx < axis34LpCnt; axis34LpIdx++) {
            MaskReg mask = UpdateMask<RT>(iAxis34Size);
            for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(innerAxis1); axis1Idx++) {
                auto aregI = Reg::CreateAddrReg<RT>(axis34LpIdx, axis34Offset, axis1Idx, axis1OutOffset);
                DataCopy(tmpIn, reOutAddr, aregI);
                for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2 - 1); axis2Idx++) {
                    auto aregO = Reg::CreateAddrReg<RT>(axis34LpIdx, axis34Offset, axis1Idx, axis1OutOffset, axis2Idx,
                                                        axis2Offset);
                    DataCopy(reOutAddr + axis2Offset, tmpIn, aregO, mask);
                }
            }
        }
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFInnerBroadcastToB(LocalTensor<T>& outputTensor,
                                                                                LocalTensor<T>& inputTensor)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)inputTensor.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)outputTensor.GetPhyAddr();

    if constexpr (sizeof(T) == sizeof(RT)) {
        VFBroadcastOneElemLB(inputAddr, outputAddr);
    } else {
        VFBroadcastOneElemLBB64(inputAddr, outputAddr);
    }

    if (innerAxis2 > 1) {
        PipeBarrier<PIPE_V>();
        VFBroadcastOneElemOB(outputAddr);
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFInnerBroadcastToA(LocalTensor<T>& outputTensor,
                                                                                LocalTensor<T>& inputTensor)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)inputTensor.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)outputTensor.GetPhyAddr();
    auto reInAddr = reinterpret_cast<__local_mem__ RT*>(inputAddr);
    auto reOutAddr = reinterpret_cast<__local_mem__ RT*>(outputAddr);

    uint32_t axis4BA = Ops::Base::CeilAlign(innerAxis4, dataAlignCnt);
    uint32_t axis4Offset = VL_CNT;
    if constexpr (sizeof(T) != sizeof(RT)) {
        axis4BA *= nTwo;
        axis4Offset *= nTwo;
    }
    uint32_t axis2InOffset = axis4BA;
    uint32_t axis1Offset = innerAxis2 * innerAxis3 * axis4BA;
    uint32_t axis2Offset = innerAxis3 * axis4BA;
    uint32_t axis3Offset = axis4BA;
    uint32_t lastASize = axis4BA;
    uint16_t axis4LpCnt = Ops::Base::CeilDiv(innerAxis4, VL_CNT);
    // BABA
    if (innerAxis1 != 1) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> tmpIn;
            for (uint16_t axis4LpIdx = 0; axis4LpIdx < axis4LpCnt; axis4LpIdx++) {
                MaskReg mask = UpdateMask<RT>(lastASize);
                for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2); axis2Idx++) {
                    auto aregI = Reg::CreateAddrReg<RT>(axis4LpIdx, axis4Offset, axis2Idx, axis2InOffset);
                    DataCopy(tmpIn, reInAddr, aregI);
                    for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(innerAxis1); axis1Idx++) {
                        for (uint16_t axis3Idx = 0; axis3Idx < static_cast<uint16_t>(innerAxis3); axis3Idx++) {
                            auto aregO = Reg::CreateAddrReg<RT>(axis4LpIdx, axis4Offset, axis2Idx, axis2Offset,
                                                                axis1Idx, axis1Offset, axis3Idx, axis3Offset);
                            DataCopy(reOutAddr, tmpIn, aregO, mask);
                        }
                    }
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> tmpIn;
            for (uint16_t axis4LpIdx = 0; axis4LpIdx < axis4LpCnt; axis4LpIdx++) {
                MaskReg mask = UpdateMask<RT>(lastASize);
                for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2); axis2Idx++) {
                    auto aregI = Reg::CreateAddrReg<RT>(axis4LpIdx, axis4Offset, axis2Idx, axis2InOffset);
                    DataCopy(tmpIn, reInAddr, aregI);
                    for (uint16_t axis3Idx = 0; axis3Idx < static_cast<uint16_t>(innerAxis3); axis3Idx++) {
                        auto aregO = Reg::CreateAddrReg<RT>(axis4LpIdx, axis4Offset, axis2Idx, axis2Offset, axis3Idx,
                                                            axis3Offset);
                        DataCopy(reOutAddr, tmpIn, aregO, mask);
                    }
                }
            }
        }
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFInnerBrcLastDimLEBlock(LocalTensor<T>& outputTensor,
                                                                                     LocalTensor<T>& inputTensor)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)inputTensor.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)outputTensor.GetPhyAddr();
    auto reInAddr = reinterpret_cast<__local_mem__ RT*>(inputAddr);
    auto reOutAddr = reinterpret_cast<__local_mem__ RT*>(outputAddr);

    uint32_t axis4BA = Ops::Base::CeilAlign(innerAxis4, dataAlignCnt);
    if constexpr (sizeof(T) != sizeof(RT)) {
        axis4BA *= nTwo;
    }
    uint32_t axis2InOffset = axis4BA;
    uint32_t axis2Offset = innerAxis3 * axis4BA;
    uint32_t axis1Offset = innerAxis2 * axis2Offset;
    uint32_t lastASize = axis2Offset;
    // BABA
    if (innerAxis1 != 1) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> tmpIn;
            MaskReg mask = UpdateMask<RT>(lastASize);
            for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2); axis2Idx++) {
                DataCopy<RT, Reg::LoadDist::DIST_BLK>(tmpIn, reInAddr + axis2Idx * axis2InOffset);
                for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(innerAxis1); axis1Idx++) {
                    DataCopy(reOutAddr + axis2Idx * axis2Offset + axis1Idx * axis1Offset, tmpIn, mask);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> tmpIn;
            MaskReg mask = UpdateMask<RT>(lastASize);
            for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2); axis2Idx++) {
                DataCopy<RT, Reg::LoadDist::DIST_BLK>(tmpIn, reInAddr + axis2Idx * axis2InOffset);
                DataCopy(reOutAddr + axis2Idx * axis2Offset, tmpIn, mask);
            }
        }
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::GenGatherIdx(Reg::RegTensor<int32_t>& gatherIdx,
                                                                         int32_t axis4BA)
{
    Reg::MaskReg mask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<int32_t> axis4Reg;
    Reg::RegTensor<int32_t> divReg;
    Reg::Duplicate(axis4Reg, axis4BA);
    Reg::Arange(gatherIdx, 0);
    Reg::Div(divReg, gatherIdx, axis4Reg, mask);
    Reg::Mul(axis4Reg, axis4Reg, divReg, mask);
    Reg::Sub(gatherIdx, gatherIdx, axis4Reg, mask);
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::VFInnerBrcLastDimGTBlock(LocalTensor<T>& outputTensor,
                                                                                     LocalTensor<T>& inputTensor)
{
    __local_mem__ T* inputAddr = (__local_mem__ T*)inputTensor.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)outputTensor.GetPhyAddr();
    auto reInAddr = reinterpret_cast<__local_mem__ int32_t*>(inputAddr);
    auto reOutAddr = reinterpret_cast<__local_mem__ int32_t*>(outputAddr);

    uint32_t axis4BA = Ops::Base::CeilAlign(innerAxis4, dataAlignCnt);
    if constexpr (sizeof(T) == sizeof(int64_t)) {
        axis4BA *= nTwo;
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        axis4BA /= nTwo;
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        axis4BA /= (nTwo * nTwo);
    }
    uint32_t axis2InOffset = axis4BA;
    uint32_t axis2Offset = innerAxis3 * axis4BA;
    uint32_t axis1Offset = innerAxis2 * axis2Offset;
    uint32_t lastASize = axis2Offset;
    // BABA
    if (innerAxis1 != 1) {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> tmpIn;
            AscendC::Reg::RegTensor<int32_t> tmpOut;
            AscendC::Reg::RegTensor<int32_t> gatherIdx;
            GenGatherIdx(gatherIdx, static_cast<int32_t>(axis4BA));
            MaskReg mask = UpdateMask<int32_t>(lastASize);
            for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2); axis2Idx++) {
                DataCopy(tmpIn, reInAddr + axis2Idx * axis2InOffset);
                Reg::Gather(tmpOut, tmpIn, (Reg::RegTensor<uint32_t>&)gatherIdx);
                for (uint16_t axis1Idx = 0; axis1Idx < static_cast<uint16_t>(innerAxis1); axis1Idx++) {
                    DataCopy(reOutAddr + axis2Idx * axis2Offset + axis1Idx * axis1Offset, tmpOut, mask);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> tmpIn;
            AscendC::Reg::RegTensor<int32_t> tmpOut;
            AscendC::Reg::RegTensor<int32_t> gatherIdx;
            GenGatherIdx(gatherIdx, static_cast<int32_t>(axis4BA));
            MaskReg mask = UpdateMask<int32_t>(lastASize);
            for (uint16_t axis2Idx = 0; axis2Idx < static_cast<uint16_t>(innerAxis2); axis2Idx++) {
                DataCopy(tmpIn, reInAddr + axis2Idx * axis2InOffset);
                Reg::Gather(tmpOut, tmpIn, (Reg::RegTensor<uint32_t>&)gatherIdx);
                DataCopy(reOutAddr + axis2Idx * axis2Offset, tmpOut, mask);
            }
        }
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::BroadcastUb(LocalTensor<T>& outputTensor)
{
    innerAxis0 = innerShape[0];
    innerAxis1 = innerShape[1];
    innerAxis2 = innerShape[DIM_TWO];
    innerAxis3 = innerShape[DIM_THREE];
    innerAxis4 = innerShape[DIM_FOUR];
    LocalTensor<T> inputTensor = inQueue_.DeQue<T>();
    if constexpr (!isLastDimSmall) {
        if (tdPtr_->isLastDimB == 1) {
            VFInnerBroadcastToB(outputTensor, inputTensor);
        } else {
            VFInnerBroadcastToA(outputTensor, inputTensor);
        }
    } else {
        if (innerAxis4 <= dataAlignCnt) {
            VFInnerBrcLastDimLEBlock(outputTensor, inputTensor);
        } else {
            VFInnerBrcLastDimGTBlock(outputTensor, inputTensor);
        }
    }
    inQueue_.FreeTensor(inputTensor);
    outQueue_.EnQue(outputTensor);
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::BrcUIsB(int64_t aIdx)
{
    innerShape[MAX_INNER_DIM - tdPtr_->uAxisCnt] = tdPtr_->uLpUnit;
    SetInputParams(1);
    SetOutputParams(tdPtr_->uLpUnit);

    LocalTensor<T> inTensor = inQueue_.AllocTensor<T>();
    LocalTensor<T> outTensor = outQueue_.AllocTensor<T>();

    gmInOffset = (this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx) + inBlockOffset);
    CopyDataIn(inTensor);
    BroadcastUb(outTensor);
    LocalTensor<T> broTensor = outQueue_.DeQue<T>();

    int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx);
    for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
        gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
        CopyDataOut(broTensor);
    }
    if (lpInfo.uLeft > 0) {
        SetOutputParams(lpInfo.uLeft);
        gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
        CopyDataOut(broTensor);
    }
    outQueue_.FreeTensor(broTensor);
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::BrcUNotB(int64_t aIdx)
{
    innerShape[MAX_INNER_DIM - tdPtr_->uAxisCnt] = tdPtr_->uLpUnit;
    SetInputParams(tdPtr_->uLpUnit);
    SetOutputParams(tdPtr_->uLpUnit);

    int64_t aInOffset = this->CalcAInAxesOffset(tdPtr_, aBaseIdx + aIdx);
    int64_t aOutOffset = this->CalcAOutAxesOffset(tdPtr_, aBaseIdx + aIdx);
    for (int64_t uLpIdx = lpInfo.uLpBegIdx; uLpIdx < lpInfo.uLpEndIdx; uLpIdx++) {
        LocalTensor<T> inTensor = inQueue_.AllocTensor<T>();
        LocalTensor<T> outTensor = outQueue_.AllocTensor<T>();
        gmInOffset = (aInOffset + inBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uInOffset);
        CopyDataIn(inTensor);
        BroadcastUb(outTensor);
        LocalTensor<T> broTensor = outQueue_.DeQue<T>();
        gmOutOffset = (aOutOffset + outBlockOffset + uLpIdx * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
        CopyDataOut(broTensor);
        outQueue_.FreeTensor(broTensor);
    }
    if (lpInfo.uLeft > 0) {
        innerShape[MAX_INNER_DIM - tdPtr_->uAxisCnt] = lpInfo.uLeft;
        SetInputParams(lpInfo.uLeft);
        SetOutputParams(lpInfo.uLeft);
        LocalTensor<T> inTensor = inQueue_.AllocTensor<T>();
        LocalTensor<T> outTensor = outQueue_.AllocTensor<T>();
        gmInOffset = (aInOffset + inBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uInOffset);
        CopyDataIn(inTensor);
        BroadcastUb(outTensor);
        LocalTensor<T> broTensor = outQueue_.DeQue<T>();
        gmOutOffset = (aOutOffset + outBlockOffset + lpInfo.uLpCnt * tdPtr_->uLpUnit * tdPtr_->uOutOffset);
        CopyDataOut(broTensor);
        outQueue_.FreeTensor(broTensor);
    }
}

template <typename T, typename U, bool isLastDimSmall>
__aicore__ inline void BroadcastToUb<T, U, isLastDimSmall>::Process()
{
    this->CalcABBaseIdx(tdPtr_, blockIdx, aBaseIdx, bBaseIdx);
    this->CalcAxesLoopInfo(tdPtr_, blockIdx, lpInfo);
    this->CalcInBlockOffset(tdPtr_, blockIdx, inBlockOffset);
    this->CalcOutBlockOffset(tdPtr_, blockIdx, outBlockOffset);

    CalcInnerShape();
    for (int64_t aIdx = 0; aIdx < lpInfo.aLpCnt; aIdx++) {
        if (tdPtr_->isUNotB == 0) {
            BrcUIsB(aIdx);
        } else {
            BrcUNotB(aIdx);
        }
    }
}

} // namespace BrcTo

#endif // BROADCAST_TO_WITH_UB_H_
