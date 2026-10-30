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
 * \file roll_gather_simd.h
 * \brief
 */

#ifndef __ROLL_GATHER_SIMD_H__
#define __ROLL_GATHER_SIMD_H__

#include "kernel_operator.h"
#include "roll_struct.h"
#include "op_kernel/platform_util.h"

namespace Roll {
using namespace AscendC;
constexpr int32_t GATHER_REG_SIZE = Ops::Base::GetVRegSize();
template <typename T, bool isShiftW = true>
class RollGatherSimd {
public:
    __aicore__ inline RollGatherSimd(TPipe* pipe, const RollTilingData* tiling) : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyIn(int64_t index, int32_t len);
    template <typename IntType, typename IndexType, bool isShiftH>
    __aicore__ inline void CalculateFullCHWIndex(__ubuf__ IndexType* indexLocalAddr, uint16_t& tmplSize);
    template <typename IntType, typename IndexType>
    __aicore__ inline void CalculateFullHWIndex(__ubuf__ IndexType* indexLocalAddr, uint16_t& tmplSize);
    template <typename IntType, typename IndexType>
    __aicore__ inline void CalculateFullWIndex(__ubuf__ IndexType* indexLocalAddr, uint16_t& tmplSize);
    __aicore__ inline void findTargetSplit(uint16_t& gatherTmplsplit, bool& isEqual);
    __aicore__ inline void CopyOut(int64_t outindex, int32_t Count);
    __aicore__ inline void ComputeOutIndex(int64_t inputIndex, int64_t& outputIndex, uint16_t& gatherTmplsplit,
                                           int32_t& Count);
    __aicore__ inline void Process();

private:
    template <typename IndexType>
    __aicore__ inline void Gather(LocalTensor<T>& xTensor, LocalTensor<T>& gatherTmpl, int32_t Count, uint16_t tmplSize,
                                  int32_t addrShift);

private:
    const RollTilingData* tilingData_;
    TPipe* pipe_;
    UbParam curCoreUbParam;
    TQueBind<TPosition::GM, TPosition::VECIN, 1> xInQue_;
    TQueBind<TPosition::VECOUT, TPosition::GM, 1> AlienBuf;
    TBuf<TPosition::VECCALC> gatherMethodBuf;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int32_t hwLen;
    int64_t curCoreBaseIndex_ = 0;
    int64_t blockIdx_ = 0;
    int64_t inputIndices_[MAX_DIM_NUM] = {0};
    int32_t gatherKey = 10001;
};

template <typename T, bool isShiftW>
__aicore__ inline void RollGatherSimd<T, isShiftW>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreUbParam = tilingData_->tailCoreUbParam;
        if (curCoreUbParam.UbFactor == 0) {
            curCoreUbParam.UbFactor = curCoreUbParam.UbTailFactor;
        }
        curCoreUbParam.UbFactor *= tilingData_->strides[curCoreUbParam.UbSplitAxis];
        curCoreUbParam.UbCount = (tilingData_->blockTailFactor * tilingData_->strides[tilingData_->blockSplitAxis] +
                                  curCoreUbParam.UbFactor - 1) /
                                 curCoreUbParam.UbFactor;
        curCoreUbParam.UbTailFactor = tilingData_->blockTailFactor * tilingData_->strides[tilingData_->blockSplitAxis] -
                                      (curCoreUbParam.UbCount - 1) * curCoreUbParam.UbFactor;
    } else {
        curCoreUbParam = tilingData_->mainCoreUbParam;
        if (curCoreUbParam.UbFactor == 0) {
            curCoreUbParam.UbFactor = curCoreUbParam.UbTailFactor;
        }
        curCoreUbParam.UbFactor *= tilingData_->strides[curCoreUbParam.UbSplitAxis];
        curCoreUbParam.UbCount = (tilingData_->blockFactor * tilingData_->strides[tilingData_->blockSplitAxis] +
                                  curCoreUbParam.UbFactor - 1) /
                                 curCoreUbParam.UbFactor;
        curCoreUbParam.UbTailFactor = tilingData_->blockFactor * tilingData_->strides[tilingData_->blockSplitAxis] -
                                      (curCoreUbParam.UbCount - 1) * curCoreUbParam.UbFactor;
    }
    curCoreBaseIndex_ = tilingData_->blockFactor * blockIdx_ * tilingData_->strides[tilingData_->blockSplitAxis];
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    if (tilingData_->dimNum < 2) {
        hwLen = tilingData_->shapes[tilingData_->dimNum - 1];
    } else {
        hwLen = tilingData_->shapes[tilingData_->dimNum - 2] * tilingData_->shapes[tilingData_->dimNum - 1];
    }

    pipe_->InitBuffer(xInQue_, BUF_NUM, curCoreUbParam.UbFactor * sizeof(T) + GATHER_REG_SIZE);
    pipe_->InitBuffer(AlienBuf, BUF_NUM, curCoreUbParam.UbFactor * sizeof(T) + GATHER_REG_SIZE);
    pipe_->InitBuffer(gatherMethodBuf, GATHER_REG_SIZE);
}

template <typename T, bool isShiftW>
__aicore__ inline void RollGatherSimd<T, isShiftW>::CopyIn(int64_t index, int32_t len)
{
    LocalTensor<T> xTensor = xInQue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = len * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xTensor, xGm_[index], dataCopyParams, dataCopyPadParams);
    xInQue_.EnQue<T>(xTensor);
}

template <typename T, bool isShiftW>
__aicore__ inline void RollGatherSimd<T, isShiftW>::CopyOut(int64_t outindex, int32_t Count)
{
    LocalTensor<T> AlienTensor = AlienBuf.DeQue<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = Count * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(yGm_[outindex], AlienTensor, dataCopyParams);
    AlienBuf.FreeTensor<T>(AlienTensor);
}

template <typename T, bool isShiftW>
__aicore__ inline void RollGatherSimd<T, isShiftW>::Process()
{
    LocalTensor<T> gatherTmpl = gatherMethodBuf.Get<T>();
    uint16_t tmplSize;
    uint16_t gatherTmplsplit;
    uint16_t is_b8 = 0;
    if constexpr (sizeof(T) == 1) {
        is_b8 = 1;
    }
    if ((tilingData_->dimNum > 2) &&
        (tilingData_->shapes[tilingData_->dimNum - 3] * hwLen * sizeof(T) * (is_b8 + 1) <= 256) &&
        (tilingData_->shifts[tilingData_->dimNum - 3] != 0)) {
        gatherTmplsplit = tilingData_->dimNum - 3;
        if constexpr (sizeof(T) <= 2) {
            if (tilingData_->shifts[tilingData_->dimNum - 2] == 0) {
                CalculateFullCHWIndex<int16_t, uint16_t, false>((__ubuf__ uint16_t*)gatherTmpl.GetPhyAddr(), tmplSize);
            } else {
                CalculateFullCHWIndex<int16_t, uint16_t, true>((__ubuf__ uint16_t*)gatherTmpl.GetPhyAddr(), tmplSize);
            }
        } else {
            if (tilingData_->shifts[tilingData_->dimNum - 2] == 0) {
                CalculateFullCHWIndex<int32_t, uint32_t, false>((__ubuf__ uint32_t*)gatherTmpl.GetPhyAddr(), tmplSize);
            } else {
                CalculateFullCHWIndex<int32_t, uint32_t, true>((__ubuf__ uint32_t*)gatherTmpl.GetPhyAddr(), tmplSize);
            }
        }
    } else if (hwLen * sizeof(T) * (is_b8 + 1) > GATHER_REG_SIZE || tilingData_->dimNum < 2) {
        gatherTmplsplit = tilingData_->dimNum - 1;
        if constexpr (sizeof(T) <= 2) {
            CalculateFullWIndex<int16_t, uint16_t>((__ubuf__ uint16_t*)gatherTmpl.GetPhyAddr(), tmplSize);
        } else {
            CalculateFullWIndex<int32_t, uint32_t>((__ubuf__ uint32_t*)gatherTmpl.GetPhyAddr(), tmplSize);
        }
    } else {
        gatherTmplsplit = tilingData_->dimNum - 2;
        if constexpr (sizeof(T) <= 2) {
            CalculateFullHWIndex<int16_t, uint16_t>((__ubuf__ uint16_t*)gatherTmpl.GetPhyAddr(), tmplSize);
        } else {
            CalculateFullHWIndex<int32_t, uint32_t>((__ubuf__ uint32_t*)gatherTmpl.GetPhyAddr(), tmplSize);
        }
    }
    bool isEqual = false;
    uint16_t movTmplsplit = gatherTmplsplit;
    findTargetSplit(movTmplsplit, isEqual);
    for (int32_t loopNum = 0; loopNum < curCoreUbParam.UbCount; loopNum++) {
        int32_t curUbFactor;
        if (loopNum == curCoreUbParam.UbCount - 1) {
            curUbFactor = curCoreUbParam.UbTailFactor;
        } else {
            curUbFactor = curCoreUbParam.UbFactor;
        }
        int32_t CopyLoop = 1;
        int32_t loopSize = curUbFactor;
        if (!isEqual) {
            CopyLoop = curUbFactor /
                       (tilingData_->strides[movTmplsplit] * tilingData_->shapes[movTmplsplit]); // 必然整除
            loopSize = tilingData_->strides[movTmplsplit - 1];
        }
        CopyIn(curCoreBaseIndex_, curUbFactor);
        LocalTensor<T> xTensor = xInQue_.DeQue<T>();
        for (uint16_t i = 0; i < CopyLoop; i++) {
            int32_t Count = 0;
            int32_t countLoopSize = loopSize;
            int32_t addrShift = i * loopSize;
            while (countLoopSize > 0) {
                int64_t outIndex = 0;
                if (gatherTmplsplit != movTmplsplit) {
                    ComputeOutIndex(curCoreBaseIndex_ + addrShift, outIndex, movTmplsplit, Count);
                } else {
                    Count = loopSize;
                }
                if (countLoopSize < Count) {
                    Count = countLoopSize;
                }
                if constexpr (sizeof(T) <= 2) {
                    Gather<uint16_t>(xTensor, gatherTmpl, Count, tmplSize, addrShift);
                } else {
                    Gather<uint32_t>(xTensor, gatherTmpl, Count, tmplSize, addrShift);
                }
                CopyOut(outIndex, Count);
                countLoopSize -= Count;
                addrShift += Count;
            }
        }
        xInQue_.FreeTensor<T>(xTensor);
        curCoreBaseIndex_ += curCoreUbParam.UbFactor;
    }
    gatherMethodBuf.FreeTensor<T>(gatherTmpl);
}

template <typename T, bool isShiftW>
template <typename IntType, typename IndexType, bool isShiftH>
__aicore__ inline void RollGatherSimd<T, isShiftW>::CalculateFullCHWIndex(__ubuf__ IndexType* indexLocalAddr,
                                                                          uint16_t& tmplSize)
{
    int32_t shapesW = tilingData_->shapes[tilingData_->dimNum - 1];
    int32_t shapesH = tilingData_->shapes[tilingData_->dimNum - 2];
    int32_t shapesC = tilingData_->shapes[tilingData_->dimNum - 3];
    int32_t shiftsW = tilingData_->shifts[tilingData_->dimNum - 1];
    int32_t shiftsH = tilingData_->shifts[tilingData_->dimNum - 2];
    int32_t shiftsC = tilingData_->shifts[tilingData_->dimNum - 3];

    int32_t shapesLen = shapesW * shapesH * shapesC;

    uint16_t loopSize = 0;
    if constexpr (sizeof(T) == 1) {
        loopSize = GATHER_REG_SIZE / sizeof(T) / 2 / shapesLen;
    } else {
        loopSize = GATHER_REG_SIZE / sizeof(T) / shapesLen;
    }
    tmplSize = loopSize * shapesLen;
    IndexType copyLenth = shapesLen;
    IntType startScalar = 0;
    IntType negShiftWScalar = -1 * static_cast<IntType>(shiftsW);
    IntType wScalar = static_cast<IntType>(shapesW);
    IntType negShiftHWScalar = -1 * static_cast<IntType>(shapesW) * static_cast<IntType>(shiftsH);
    IntType hwScalar = static_cast<IntType>(shapesW) * static_cast<IntType>(shapesH);
    IntType negShiftCHWScalar = -1 * hwScalar * static_cast<IntType>(shiftsC);
    IntType chwScalar = static_cast<IntType>(shapesLen);
    uint32_t shiftCHWSizeScalar = static_cast<uint32_t>(hwScalar) * static_cast<uint32_t>(shiftsC);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<IndexType> indexVReg;
        AscendC::Reg::RegTensor<IntType> calculateVReg, helpCmpVReg, helpAddVReg, helpDivVReg;
        AscendC::Reg::MaskReg cmpResultMaskReg, helpAddCHWMaskReg, fullMaskReg;
        AscendC::Reg::UnalignReg u0;

        fullMaskReg = AscendC::Reg::CreateMask<IndexType, AscendC::Reg::MaskPattern::ALL>();
        helpAddCHWMaskReg = AscendC::Reg::UpdateMask<IndexType>(shiftCHWSizeScalar);
        AscendC::Reg::Duplicate(helpDivVReg, wScalar, fullMaskReg);
        AscendC::Reg::Arange(calculateVReg, startScalar);
        if constexpr (isShiftW) {
            AscendC::Reg::Div(helpCmpVReg, calculateVReg, helpDivVReg, fullMaskReg);
            AscendC::Reg::Muls(helpCmpVReg, helpCmpVReg, wScalar, fullMaskReg);
            AscendC::Reg::Adds(calculateVReg, calculateVReg, negShiftWScalar, fullMaskReg);
            AscendC::Reg::Compare<IntType, CMPMODE::LT>(cmpResultMaskReg, calculateVReg, helpCmpVReg, fullMaskReg);
            AscendC::Reg::Duplicate(helpAddVReg, wScalar, cmpResultMaskReg);
            AscendC::Reg::Add(calculateVReg, calculateVReg, helpAddVReg, fullMaskReg);
        }
        if constexpr (isShiftH) {
            AscendC::Reg::Duplicate(helpDivVReg, hwScalar, fullMaskReg);
            AscendC::Reg::Div(helpCmpVReg, calculateVReg, helpDivVReg, fullMaskReg);
            AscendC::Reg::Muls(helpCmpVReg, helpCmpVReg, hwScalar, fullMaskReg);
            AscendC::Reg::Adds(calculateVReg, calculateVReg, negShiftHWScalar, fullMaskReg);
            AscendC::Reg::Compare<IntType, CMPMODE::LT>(cmpResultMaskReg, calculateVReg, helpCmpVReg, fullMaskReg);
            AscendC::Reg::Duplicate(helpAddVReg, hwScalar, cmpResultMaskReg);
            AscendC::Reg::Add(calculateVReg, calculateVReg, helpAddVReg, fullMaskReg);
        }
        AscendC::Reg::Adds(calculateVReg, calculateVReg, negShiftCHWScalar, fullMaskReg);
        AscendC::Reg::Duplicate(helpAddVReg, shapesLen, helpAddCHWMaskReg);
        AscendC::Reg::Add(calculateVReg, calculateVReg, helpAddVReg, fullMaskReg);
        AscendC::Reg::Copy(indexVReg, (Reg::RegTensor<IndexType>&)calculateVReg);
        AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth);
        for (uint16_t i = 1; i < loopSize; i++) {
            AscendC::Reg::Adds(indexVReg, indexVReg, copyLenth, fullMaskReg);
            AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth);
        }
        AscendC::Reg::DataCopyUnAlignPost(indexLocalAddr, u0, 0);
    }
}

template <typename T, bool isShiftW>
template <typename IntType, typename IndexType>
__aicore__ inline void RollGatherSimd<T, isShiftW>::CalculateFullHWIndex(__ubuf__ IndexType* indexLocalAddr,
                                                                         uint16_t& tmplSize)
{
    int32_t shapesW = tilingData_->shapes[tilingData_->dimNum - 1];
    int32_t shapseH = tilingData_->shapes[tilingData_->dimNum - 2];
    int32_t shiftsW = tilingData_->shifts[tilingData_->dimNum - 1];
    int32_t shiftsH = tilingData_->shifts[tilingData_->dimNum - 2];

    uint16_t loopSize = GATHER_REG_SIZE / sizeof(T) / hwLen;
    if constexpr (sizeof(T) == 1) {
        loopSize = GATHER_REG_SIZE / sizeof(T) / 2 / hwLen;
    }
    tmplSize = loopSize * hwLen;
    IndexType copyLenth = hwLen;
    IntType startScalar = 0;
    IntType negShiftWScalar = -1 * static_cast<IntType>(shiftsW);
    IntType wScalar = static_cast<IntType>(shapesW);
    IntType negShiftHWScalar = -1 * static_cast<IntType>(shiftsH) * static_cast<IntType>(shapesW);
    IntType HWScalar = static_cast<IntType>(hwLen);
    uint32_t shifthWSizeScalar = static_cast<uint32_t>(shiftsH) * static_cast<uint32_t>(shapesW);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<IndexType> indexVReg;
        AscendC::Reg::RegTensor<IntType> calculateVReg, helpCmpVReg, helpAddVReg, helpDivVReg;
        AscendC::Reg::MaskReg cmpResultMaskReg, helpAddHWMaskReg, fullMaskReg;
        AscendC::Reg::UnalignReg u0;

        fullMaskReg = AscendC::Reg::CreateMask<IndexType, AscendC::Reg::MaskPattern::ALL>();
        helpAddHWMaskReg = AscendC::Reg::UpdateMask<IndexType>(shifthWSizeScalar);
        AscendC::Reg::Duplicate(helpDivVReg, wScalar, fullMaskReg);
        AscendC::Reg::Arange(calculateVReg, startScalar);
        if constexpr (isShiftW) {
            AscendC::Reg::Div(helpCmpVReg, calculateVReg, helpDivVReg, fullMaskReg);
            AscendC::Reg::Muls(helpCmpVReg, helpCmpVReg, wScalar, fullMaskReg);
            AscendC::Reg::Adds(calculateVReg, calculateVReg, negShiftWScalar, fullMaskReg);
            AscendC::Reg::Compare<IntType, CMPMODE::LT>(cmpResultMaskReg, calculateVReg, helpCmpVReg, fullMaskReg);
            AscendC::Reg::Duplicate(helpAddVReg, wScalar, cmpResultMaskReg);
            AscendC::Reg::Add(calculateVReg, calculateVReg, helpAddVReg, fullMaskReg);
        }
        AscendC::Reg::Adds(calculateVReg, calculateVReg, negShiftHWScalar, fullMaskReg);
        AscendC::Reg::Duplicate(helpAddVReg, hwLen, helpAddHWMaskReg);
        AscendC::Reg::Add(calculateVReg, calculateVReg, helpAddVReg, fullMaskReg);
        AscendC::Reg::Copy(indexVReg, (Reg::RegTensor<IndexType>&)calculateVReg);
        AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth);
        for (uint16_t i = 1; i < loopSize; i++) {
            AscendC::Reg::Adds(indexVReg, indexVReg, copyLenth, fullMaskReg);
            AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth);
        }
        AscendC::Reg::DataCopyUnAlignPost(indexLocalAddr, u0, 0);
    }
}
template <typename T, bool isShiftW>
template <typename IntType, typename IndexType>
__aicore__ inline void RollGatherSimd<T, isShiftW>::CalculateFullWIndex(__ubuf__ IndexType* indexLocalAddr,
                                                                        uint16_t& tmplSize)
{
    int64_t shapesW = tilingData_->shapes[tilingData_->dimNum - 1];
    int64_t shiftsW = tilingData_->shifts[tilingData_->dimNum - 1];
    uint16_t wSize = static_cast<uint16_t>(shapesW);
    uint16_t loopSize = GATHER_REG_SIZE / sizeof(T) / wSize;
    if constexpr (sizeof(T) == 1) {
        loopSize = GATHER_REG_SIZE / sizeof(T) / 2 / wSize;
    }
    tmplSize = loopSize * wSize;
    IndexType copyLenth = wSize;
    IntType startScalar = 0;
    IntType negShiftWScalar = -1 * static_cast<IntType>(shiftsW);
    IntType wScalar = static_cast<IntType>(shapesW);
    uint32_t shiftwSizeScalar = static_cast<uint32_t>(shiftsW);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<IndexType> indexVReg;
        AscendC::Reg::RegTensor<IntType> calculateVReg, helpAddVReg;
        AscendC::Reg::MaskReg helpAddHWMaskReg, fullMaskReg;
        AscendC::Reg::UnalignReg u0;

        fullMaskReg = AscendC::Reg::CreateMask<IndexType, AscendC::Reg::MaskPattern::ALL>();
        helpAddHWMaskReg = AscendC::Reg::UpdateMask<IndexType>(shiftwSizeScalar);

        AscendC::Reg::Arange(calculateVReg, startScalar);
        if constexpr (isShiftW) {
            AscendC::Reg::Adds(calculateVReg, calculateVReg, negShiftWScalar, fullMaskReg);
            AscendC::Reg::Duplicate(helpAddVReg, wScalar, helpAddHWMaskReg);
            AscendC::Reg::Add(calculateVReg, calculateVReg, helpAddVReg, fullMaskReg);
        }
        AscendC::Reg::Copy(indexVReg, (Reg::RegTensor<IndexType>&)calculateVReg);
        if constexpr (isShiftW) {
            AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth);
            for (uint16_t i = 1; i < loopSize; i++) {
                AscendC::Reg::Adds(indexVReg, indexVReg, copyLenth, fullMaskReg);
                AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth);
            }
        } else {
            AscendC::Reg::DataCopyUnAlign(indexLocalAddr, indexVReg, u0, copyLenth * loopSize);
        }
        AscendC::Reg::DataCopyUnAlignPost(indexLocalAddr, u0, 0);
    }
}
template <typename T, bool isShiftW>
__aicore__ inline void RollGatherSimd<T, isShiftW>::findTargetSplit(uint16_t& movTmplsplit, bool& isEqual)
{
    for (int32_t i = movTmplsplit - 1; i >= curCoreUbParam.UbSplitAxis; i--) {
        if (tilingData_->shifts[i] != 0) {
            movTmplsplit = static_cast<uint16_t>(i);
            isEqual = (i == curCoreUbParam.UbSplitAxis);
            return;
        }
    }
    movTmplsplit = static_cast<uint16_t>(curCoreUbParam.UbSplitAxis);
    isEqual = true;
    return;
}

template <typename T, bool isShiftW>
__aicore__ inline void RollGatherSimd<T, isShiftW>::ComputeOutIndex(int64_t inputIndex, int64_t& outputIndex,
                                                                    uint16_t& movTmplsplit,
                                                                    int32_t& Count) //, int64_t& count)
{
    for (int64_t dim = 0; dim < tilingData_->dimNum; dim++) {
        inputIndices_[dim] = inputIndex / tilingData_->strides[dim];
        inputIndex = inputIndex % tilingData_->strides[dim];
        if (dim < movTmplsplit + 1) {
            outputIndex += (inputIndices_[dim] + tilingData_->shifts[dim]) % tilingData_->shapes[dim] *
                           tilingData_->strides[dim];
        }
    }
    int64_t currentPos = inputIndices_[movTmplsplit];
    int64_t shiftPoint = tilingData_->shapes[movTmplsplit] - tilingData_->shifts[movTmplsplit];
    int64_t axisSize = tilingData_->shapes[movTmplsplit];

    if (currentPos < shiftPoint) {
        Count = (shiftPoint - currentPos) * tilingData_->strides[movTmplsplit];
    } else {
        Count = (axisSize - currentPos) * tilingData_->strides[movTmplsplit];
    }
}

template <typename T, bool isShiftW>
template <typename IndexType>
__aicore__ inline void RollGatherSimd<T, isShiftW>::Gather(LocalTensor<T>& xTensor, LocalTensor<T>& gatherTmpl,
                                                           int32_t Count, uint16_t tmplSize, int32_t addrShift)
{
    LocalTensor<T> AlienTensor = AlienBuf.AllocTensor<T>();
    auto dstPtr = (__ubuf__ T*)AlienTensor.GetPhyAddr();
    auto srcPtr = (__ubuf__ T*)xTensor.GetPhyAddr() + addrShift;
    auto indexPtr = (__ubuf__ IndexType*)gatherTmpl.GetPhyAddr();
    uint16_t loopSize = (Count + tmplSize - 1) / tmplSize;
    uint32_t tmpMaskNum = tmplSize;
    __VEC_SCOPE__
    {
        Reg::RegTensor<IndexType> indexReg;
        Reg::RegTensor<T> dstReg0;
        Reg::RegTensor<T> dstReg1;
        Reg::MaskReg maskReg;
        Reg::UnalignReg u0;
        if constexpr (sizeof(T) == 1) {
            maskReg = Reg::UpdateMask<uint16_t>(tmpMaskNum);
        } else {
            maskReg = Reg::UpdateMask<T>(tmpMaskNum);
        }
        Reg::DataCopy(indexReg, indexPtr);
        for (uint16_t i = 0; i < loopSize; i++) {
            if constexpr (sizeof(T) == 1) {
                Reg::DataCopyGather((Reg::RegTensor<uint16_t>&)dstReg0, srcPtr + i * tmplSize, indexReg, maskReg);
                Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)dstReg1,
                                                                       (Reg::RegTensor<uint16_t>&)dstReg0);
            } else {
                Reg::DataCopyGather(dstReg1, srcPtr + i * tmplSize, indexReg, maskReg);
            }
            Reg::DataCopyUnAlign(dstPtr, (Reg::RegTensor<T>&)dstReg1, u0, tmplSize);
        }
        AscendC::Reg::DataCopyUnAlignPost(dstPtr, u0, 0);
    }
    AlienBuf.EnQue<T>(AlienTensor);
}
} // namespace Roll
#endif // __ROLL_SIMD_H__
