/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matrix_set_diag_no_cutw_v2.h
 * \brief
 */

#ifndef ASCENDC_MATRIX_SET_DIAG_NO_CUTW_V2_H_
#define ASCENDC_MATRIX_SET_DIAG_NO_CUTW_V2_H_

#include "kernel_operator.h"
#include "matrix_set_diag_v2_tilingdata.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "matrix_set_diag_v2_simt.h"

namespace MSD {
using namespace AscendC;

template <typename T, uint8_t PROC_MODE, bool VL_MODE>
class MatrixSetDiagNoCutWV2 {
private:
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static uint32_t ALIGN_NUM = 32 / sizeof(T);
    // 特殊数据类型（int8/int64/uint64）处理时元素数量翻倍的系数
    constexpr static uint16_t ELEMENT_EXPAND_TIMES = 2;

private:
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using MaskType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), T, int32_t>;
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType_ = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    template <uint32_t N>
    using SimtSetDiagFunc = SimtNoCutTailSetDiagFunc<T, N>;
    TPipe* pipe_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> diagonalGm_;
    GlobalTensor<T> outputGm_;

    TBuf<TPosition::VECCALC> inQue_;
    TBuf<TPosition::VECCALC> diagQue_;
    TBuf<TPosition::VECCALC> indexQue_;

    int32_t vlLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    int32_t blockIdx_{0};

    // tiling params
    const MSDV2NoCutTailTilingData* tdPtr_ = nullptr;
    uint32_t coreNum_{0};
    uint64_t mergeDimSize_{0};
    uint32_t xRowNum_{0};
    uint32_t xColNum_{0};
    uint32_t diagNum_{0};
    uint32_t maxDiagLen_{0};
    uint32_t tailAxisDataSize_{0};
    uint32_t tailDiagSize_{0};
    uint32_t ubFactor_{0};
    uint32_t oneTailSize_{0};
    uint32_t oneDiagSize_{0};
    int32_t firstK_{0};
    int32_t k0_;

public:
    __aicore__ inline MatrixSetDiagNoCutWV2(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, const MSDV2NoCutTailTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tdPtr_ = tilingData;
        coreNum_ = tdPtr_->input.coreNum;
        mergeDimSize_ = tdPtr_->input.mergeDimSize;
        xRowNum_ = tdPtr_->input.xRowNum;
        xColNum_ = tdPtr_->input.xColNum;
        diagNum_ = tdPtr_->input.diagNum;
        maxDiagLen_ = tdPtr_->input.maxDiagLen;
        tailAxisDataSize_ = xRowNum_ * xColNum_;
        ubFactor_ = tdPtr_->ubFactor;
        tailDiagSize_ = diagNum_ * maxDiagLen_;
        firstK_ = tdPtr_->input.k1;
        k0_ = tdPtr_->input.k0;
        oneTailSize_ = Ops::Base::CeilAlign(ubFactor_ * tailAxisDataSize_, ALIGN_NUM);
        oneDiagSize_ = Ops::Base::CeilAlign(ubFactor_ * tailDiagSize_, ALIGN_NUM);

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        diagonalGm_.SetGlobalBuffer((__gm__ T*)diagonal);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        if constexpr (PROC_MODE < 3) {
            if constexpr (VL_MODE == 1) {
                pipe_->InitBuffer(indexQue_, vlLen_ * sizeof(CastType_));
            } else if constexpr (PROC_MODE == 2) {
                pipe_->InitBuffer(indexQue_, Ops::Base::CeilAlign(tailDiagSize_, ALIGN_NUM) * sizeof(CastType_));
            } else if constexpr (PROC_MODE == 1) {
                pipe_->InitBuffer(indexQue_, Ops::Base::CeilAlign(tailAxisDataSize_, ALIGN_NUM) * sizeof(CastType_));
            }
        }
        pipe_->InitBuffer(inQue_, BUF_NUM * oneTailSize_ * sizeof(T));
        pipe_->InitBuffer(diagQue_, BUF_NUM * oneDiagSize_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 计算当前块的起始索引
        uint32_t startIdx = blockIdx_;
        // 计算当前块的结束索引
        uint32_t endIdx = Ceil(mergeDimSize_, static_cast<uint64_t>(ubFactor_));

        uint64_t curMergeDimIdx = 0;
        uint64_t curMergeDimIdxEnd = 0;
        uint32_t inputLen = 0;
        uint32_t diagLen = 0;

        LocalTensor<RangeType_> idxLocal;
        if constexpr (PROC_MODE == 2) {
            idxLocal = indexQue_.Get<RangeType_>();
            GenScatterIndex(idxLocal, tailDiagSize_);
        } else if constexpr (PROC_MODE == 1) {
            idxLocal = indexQue_.Get<RangeType_>();
            GenGatherIndex(idxLocal, tailAxisDataSize_);
        }
        for (uint32_t idx = startIdx; idx < endIdx; idx += coreNum_) {
            LocalTensor<T> inLocal = inQue_.Get<T>();
            LocalTensor<T> diagLocal = diagQue_.Get<T>();
            uint32_t buf_idx = (idx - startIdx) / coreNum_;
            curMergeDimIdx = idx * ubFactor_;
            curMergeDimIdxEnd = min(curMergeDimIdx + static_cast<uint64_t>(ubFactor_), mergeDimSize_);
            inputLen = static_cast<uint32_t>((curMergeDimIdxEnd - curMergeDimIdx) * tailAxisDataSize_);
            diagLen = static_cast<uint32_t>(tailDiagSize_ * (curMergeDimIdxEnd - curMergeDimIdx));
            SetWaitEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2, buf_idx);
            CopyIn(inLocal[(buf_idx & 1) * oneTailSize_], diagLocal[(buf_idx & 1) * oneDiagSize_],
                   curMergeDimIdx * tailAxisDataSize_, curMergeDimIdx * tailDiagSize_, inputLen, diagLen);
            SetWaitEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V, buf_idx);
            if constexpr (PROC_MODE == 3) {
                SimtProcess(curMergeDimIdx, curMergeDimIdxEnd, inLocal[(buf_idx & 1) * oneTailSize_],
                            diagLocal[(buf_idx & 1) * oneDiagSize_]);
            } else if constexpr (PROC_MODE == 2) {
                ScatterProcess(inLocal[(buf_idx & 1) * oneTailSize_], diagLocal[(buf_idx & 1) * oneDiagSize_], idxLocal,
                               curMergeDimIdxEnd - curMergeDimIdx);
            } else {
                GatherProcess(inLocal[(buf_idx & 1) * oneTailSize_], diagLocal[(buf_idx & 1) * oneDiagSize_], idxLocal,
                              curMergeDimIdxEnd - curMergeDimIdx);
            }
            SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3, buf_idx);
            CopyOut(inLocal[(buf_idx & 1) * oneTailSize_], curMergeDimIdx * tailAxisDataSize_, inputLen);
        }
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& inLocal, const LocalTensor<T>& diagLocal,
                                  const uint64_t inputAddr, const uint64_t diagAddr, const uint32_t inputProcessNum,
                                  const uint32_t diagProcessNum)
    {
        DataCopyPadExtParams<T> inPadParams{
            false, 0, static_cast<uint8_t>(Ops::Base::CeilAlign(inputProcessNum, ALIGN_NUM) - inputProcessNum), 0};
        DataCopyPadExtParams<T> diagPadParams{
            false, 0, static_cast<uint8_t>(Ops::Base::CeilAlign(diagProcessNum, ALIGN_NUM) - diagProcessNum), 0};
        DataCopyExtParams copyInParams = {1u, static_cast<uint32_t>(inputProcessNum * sizeof(T)), 0, 0, 0};
        DataCopyExtParams diagParams = {1u, static_cast<uint32_t>(diagProcessNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(inLocal, inputGm_[inputAddr], copyInParams, inPadParams);
        DataCopyPad(diagLocal, diagonalGm_[diagAddr], diagParams, diagPadParams);
        inQue_.EnQue(inLocal);
        diagQue_.EnQue(diagLocal);
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& inLocal, const uint64_t outAddr, const uint32_t outProcessNum)
    {
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(outProcessNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], inLocal, copyOutParams);
    }

    __aicore__ inline void GenScatterIndex(const LocalTensor<RangeType_>& idxLocal, uint32_t processSize)
    {
        uint32_t vlLen = Ops::Base::GetVRegSize() / sizeof(CastType_);

        uint16_t vlPerTile = Ops::Base::CeilDiv(static_cast<uint16_t>(processSize), static_cast<uint16_t>(vlLen));
        uint16_t tilePerVl = static_cast<uint16_t>(vlLen) / processSize - 1; // processSize必大于0
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            processSize *= ELEMENT_EXPAND_TIMES;
        }
        uint16_t diagNum = diagNum_;
        RangeType_ oneDiagLen = maxDiagLen_;
        RangeType_ firstK = firstK_;
        RangeType_ step = static_cast<RangeType_>(xColNum_) + 1;
        RangeType_ xCol = xColNum_;
        RangeType_ xColOffset = -1 * xColNum_;
        RangeType_ kNegOffset = -1 * oneDiagLen;
        RangeType_ kPosOffset = oneDiagLen + 1;
        auto* idxPtr = (__ubuf__ RangeType_*)idxLocal.GetPhyAddr();
        RangeType_ dataTypeFactor = sizeof(T) > 4 ? 2 : 1;
        RangeType_ tailAxisDataSize = tailAxisDataSize_;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType_> indexReg;
            Reg::RegTensor<RangeType_> kReg;
            Reg::RegTensor<RangeType_> tempReg;
            Reg::RegTensor<RangeType_> tempReg2;
            Reg::RegTensor<RangeType_> tempReg3;
            Reg::RegTensor<RangeType_> tempReg4;
            Reg::RegTensor<RangeType_> outIdxReg;
            Reg::MaskReg mask;
            Reg::MaskReg kMask;
            Reg::MaskReg tMask;
            Reg::UnalignRegForLoad uReg;

            Reg::Duplicate(outIdxReg, -1);
            if constexpr (VL_MODE == 0) {
                RangeType_ firstValue = 0;
                Reg::Duplicate(tempReg2, oneDiagLen);
                for (uint16_t loopIdx = 0; loopIdx < vlPerTile; loopIdx++) {
                    uint32_t temp = processSize / dataTypeFactor;
                    mask = Reg::UpdateMask<RangeType_>(processSize);
                    Reg::Arange(tempReg4, firstValue);
                    Reg::Div(kReg, tempReg4, tempReg2, mask);
                    Reg::Mul(tempReg3, tempReg2, kReg, mask);
                    Reg::Sub(indexReg, tempReg4, tempReg3, mask);
                    Reg::Duplicate(tempReg3, firstK);
                    Reg::Sub(kReg, tempReg3, kReg, mask);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GT>(tMask, kReg, 0, mask);
                    Reg::Add(tempReg, indexReg, kReg, tMask);
                    Reg::Select(indexReg, tempReg, indexReg, tMask);
                    Reg::Muls(indexReg, indexReg, step, mask);
                    Reg::Muls(kReg, kReg, xColOffset, mask);
                    Reg::Add(indexReg, indexReg, kReg, mask);
                    tMask = Reg::UpdateMask<RangeType_>(temp);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, tempReg, xCol, tMask);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, 0, tMask);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, indexReg, tailAxisDataSize, tMask);
                    Reg::Select(indexReg, indexReg, outIdxReg, tMask);
                    auto* tempPtr = idxPtr + firstValue * dataTypeFactor;
                    Reg::StoreAlign(tempPtr, indexReg, mask);
                    firstValue = firstValue + vlLen;
                }
            } else {
                uint32_t temp = processSize;
                uint32_t temp2 = processSize / dataTypeFactor;
                mask = Reg::UpdateMask<RangeType_>(temp);
                Reg::Arange(indexReg, 0);
                Reg::Duplicate(kReg, firstK);
                RangeType_ firstMVal = oneDiagLen;
                for (uint16_t i = 0; i < diagNum; i++) {
                    uint32_t temp1 = firstMVal;
                    kMask = Reg::UpdateMask<RangeType_>(temp1);
                    Reg::Not(kMask, kMask, mask);
                    Reg::Adds(tempReg, indexReg, kNegOffset, kMask);
                    Reg::Select(indexReg, tempReg, indexReg, kMask);
                    Reg::Adds(tempReg, kReg, -1, kMask);
                    Reg::Select(kReg, tempReg, kReg, kMask);
                    firstMVal = firstMVal + oneDiagLen;
                }
                Reg::Compares<RangeType_, AscendC::CMPMODE::GT>(tMask, kReg, 0, mask);
                Reg::Add(tempReg, indexReg, kReg, tMask);
                Reg::Select(indexReg, tempReg, indexReg, tMask);
                Reg::Muls(indexReg, indexReg, step, mask);
                Reg::Muls(kReg, kReg, xColOffset, mask);
                Reg::Add(indexReg, indexReg, kReg, mask);
                tMask = Reg::UpdateMask<RangeType_>(temp2);
                Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, tempReg, xCol, tMask);
                Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, 0, tMask);
                Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, indexReg, tailAxisDataSize, tMask);
                Reg::Select(indexReg, indexReg, outIdxReg, tMask);
                Reg::StoreAlign(idxPtr, indexReg, mask);
                auto* tempPtr = idxPtr + processSize / dataTypeFactor;
                for (uint16_t loopIdx = 0; loopIdx < tilePerVl; loopIdx++) {
                    Reg::Adds(tempReg, indexReg, tailAxisDataSize, tMask);
                    Reg::Select(indexReg, tempReg, indexReg, tMask);
                    Reg::StoreUnAlign(tempPtr, indexReg, uReg, processSize / dataTypeFactor);
                    Reg::StoreUnAlignPost(tempPtr, uReg, 0);
                }
            }
        }
    }

    __aicore__ inline void GenGatherIndex(const LocalTensor<RangeType_>& idxLocal, uint32_t processSize)
    {
        uint32_t vlLen = Ops::Base::GetVRegSize() / sizeof(CastType_);
        uint16_t vlPerTile = Ops::Base::CeilDiv(static_cast<uint16_t>(processSize), static_cast<uint16_t>(vlLen));
        uint16_t tilePerVl = static_cast<uint16_t>(vlLen) / processSize - 1; // processSize必大于0
        if constexpr (sizeof(T) == sizeof(uint64_t)) {
            processSize *= ELEMENT_EXPAND_TIMES;
        }
        RangeType_ oneDiagLen = -1 * maxDiagLen_;
        RangeType_ oneRowLen = xColNum_;
        RangeType_ xRowNum = xRowNum_;
        RangeType_ xColOffset = -1 * xColNum_ - 1;
        RangeType_ firstKOffset = -1 * firstK_;
        auto* idxPtr = (__ubuf__ RangeType_*)idxLocal.GetPhyAddr();
        RangeType_ dataTypeFactor = sizeof(T) > 4 ? 2 : 1;
        RangeType_ tailAxisDataSize = tailDiagSize_;

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType_> indexReg;
            Reg::RegTensor<RangeType_> kReg;
            Reg::RegTensor<RangeType_> tempReg;
            Reg::RegTensor<RangeType_> tempReg2;
            Reg::RegTensor<RangeType_> tempReg3;
            Reg::RegTensor<RangeType_> tempReg4;
            Reg::RegTensor<RangeType_> outIdxReg;
            Reg::MaskReg mask;
            Reg::MaskReg kMask;
            Reg::MaskReg tMask;
            Reg::UnalignRegForLoad uReg;

            Reg::Duplicate(outIdxReg, -1);

            if constexpr (VL_MODE == 0) {
                RangeType_ firstValue = 0;
                Reg::Duplicate(tempReg2, oneRowLen);
                for (uint16_t loopIdx = 0; loopIdx < vlPerTile; loopIdx++) {
                    uint32_t temp = processSize / dataTypeFactor;
                    mask = Reg::UpdateMask<RangeType_>(processSize);

                    Reg::Arange(tempReg4, firstValue);
                    Reg::Div(indexReg, tempReg4, tempReg2, mask);
                    Reg::Mul(tempReg3, tempReg2, indexReg, mask);
                    Reg::Sub(kReg, tempReg4, tempReg3, mask);
                    Reg::Sub(kReg, kReg, indexReg, mask);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, kReg, 0, mask);
                    Reg::Add(tempReg, indexReg, kReg, tMask);
                    Reg::Select(indexReg, tempReg, indexReg, tMask);
                    Reg::Adds(kReg, kReg, firstKOffset, mask);
                    Reg::Muls(kReg, kReg, oneDiagLen, mask);
                    Reg::Add(indexReg, indexReg, kReg, mask);
                    tMask = Reg::UpdateMask<RangeType_>(temp);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, 0, tMask);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, indexReg, tailAxisDataSize, tMask);
                    Reg::Select(indexReg, indexReg, outIdxReg, tMask);
                    auto* tempPtr = idxPtr + firstValue * dataTypeFactor;
                    Reg::StoreAlign(tempPtr, indexReg, mask);
                    firstValue = firstValue + vlLen;
                }
            } else {
                uint32_t temp = processSize;
                uint32_t temp2 = processSize / dataTypeFactor;
                mask = Reg::UpdateMask<RangeType_>(temp);
                Reg::Arange(kReg, 0);
                Reg::Duplicate(indexReg, 0);
                RangeType_ firstMVal = oneRowLen;
                for (uint16_t i = 0; i < xRowNum; i++) {
                    uint32_t temp1 = firstMVal;
                    kMask = Reg::UpdateMask<RangeType_>(temp1);
                    Reg::Not(kMask, kMask, mask);
                    Reg::Adds(tempReg, kReg, xColOffset, kMask);
                    Reg::Select(kReg, tempReg, kReg, kMask);
                    Reg::Adds(tempReg, indexReg, 1, kMask);
                    Reg::Select(indexReg, tempReg, indexReg, kMask);
                    firstMVal = firstMVal + oneRowLen;
                }
                Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, kReg, 0, mask);
                Reg::Add(tempReg, indexReg, kReg, tMask);
                Reg::Select(indexReg, tempReg, indexReg, tMask);
                Reg::Adds(kReg, kReg, firstKOffset, mask);
                Reg::Muls(kReg, kReg, oneDiagLen, mask);
                Reg::Add(indexReg, indexReg, kReg, mask);
                tMask = Reg::UpdateMask<RangeType_>(temp2);
                Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, 0, tMask);
                Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, indexReg, tailAxisDataSize, tMask);
                Reg::Select(indexReg, indexReg, outIdxReg, tMask);
                Reg::StoreAlign(idxPtr, indexReg, mask);
                auto* tempPtr = idxPtr + processSize / dataTypeFactor;
                for (uint16_t loopIdx = 0; loopIdx < tilePerVl; loopIdx++) {
                    Reg::Adds(tempReg, indexReg, tailAxisDataSize, tMask);
                    Reg::Select(indexReg, tempReg, indexReg, tMask);
                    Reg::StoreUnAlign(tempPtr, indexReg, uReg, processSize / dataTypeFactor);
                    Reg::StoreUnAlignPost(tempPtr, uReg, 0);
                }
            }
        }
    }

    __aicore__ inline void SimtProcess(uint64_t startN, uint64_t endN, const LocalTensor<T>& inLocal,
                                       const LocalTensor<T>& diagLocal)
    {
        uint32_t magic0 = 0, shift0 = 0, magic1 = 0, shift1 = 0;
        GetUintDivMagicAndShift(magic0, shift0, static_cast<uint32_t>(xRowNum_ * diagNum_));
        GetUintDivMagicAndShift(magic1, shift1, static_cast<uint32_t>(diagNum_));

        uint32_t maxIterations = static_cast<uint32_t>((endN - startN) * xRowNum_ * diagNum_);
        __ubuf__ T* diagPtr = (__ubuf__ T*)(diagLocal.GetPhyAddr());
        __ubuf__ T* xPtr = (__ubuf__ T*)(inLocal.GetPhyAddr());

        SimtDispatch<SimtSetDiagFunc>(maxIterations, firstK_, diagNum_, magic0, shift0, magic1, shift1, diagPtr, xPtr,
                                      xRowNum_, xColNum_, maxDiagLen_, maxIterations, tailAxisDataSize_, tailDiagSize_);
    }

    __aicore__ inline void ScatterProcess(const LocalTensor<T>& inLocal, const LocalTensor<T>& diagLocal,
                                          const LocalTensor<RangeType_>& idxLocal, uint32_t ProcessNum)
    {
        uint32_t vlLenR = Ops::Base::GetVRegSize() / sizeof(RangeType_);
        uint32_t vlLenC = Ops::Base::GetVRegSize() / sizeof(CastType_);
        uint16_t loop1Num;
        uint16_t loop2Num;
        uint32_t tailSize;
        uint32_t oneProcessSize;
        uint16_t hasTail;
        RangeType_ offset;

        if constexpr (VL_MODE == 0) {
            oneProcessSize = tailDiagSize_;
            loop1Num = Ops::Base::CeilDiv(static_cast<uint16_t>(oneProcessSize), static_cast<uint16_t>(vlLenC));
            loop2Num = ProcessNum;
            offset = tailAxisDataSize_;
        } else {
            loop1Num = static_cast<uint16_t>(vlLenC) / (tailDiagSize_);
            oneProcessSize = static_cast<uint32_t>(loop1Num) * (tailDiagSize_);
            loop2Num = static_cast<uint16_t>(ProcessNum) / loop1Num; // loop1Num必大于0
            tailSize = (ProcessNum % static_cast<uint32_t>(loop1Num)) * (tailDiagSize_);
            hasTail = tailSize == 0 ? 0 : 1;
            offset = static_cast<RangeType_>(loop1Num) * static_cast<RangeType_>(tailAxisDataSize_);
        }
        uint32_t diagStride = oneProcessSize;
        if constexpr (sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int64_t)) {
            oneProcessSize = oneProcessSize * ELEMENT_EXPAND_TIMES;
            if constexpr (VL_MODE == 1) {
                tailSize = tailSize * ELEMENT_EXPAND_TIMES;
            }
        }
        uint32_t dataTypeFactor = sizeof(T) > 4 ? 2 : 1;

        auto* xLocalPtr = (__ubuf__ T*)inLocal.GetPhyAddr();
        auto* diagPtr = (__ubuf__ T*)diagLocal.GetPhyAddr();
        auto* idxPtr = (__ubuf__ RangeType_*)idxLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType_> indexReg;
            Reg::RegTensor<T> diagReg;
            Reg::RegTensor<T> diagCastReg;
            Reg::MaskReg mask;
            Reg::MaskReg tMask;
            Reg::MaskReg kMask;
            Reg::UnalignRegForLoad uReg;

            if constexpr (VL_MODE == 0) {
                for (uint16_t i = 0; i < loop1Num; i++) {
                    mask = Reg::UpdateMask<MaskType_>(oneProcessSize);
                    Reg::LoadAlign(indexReg, idxPtr + i * vlLenR);
                    for (uint16_t j = 0; j < loop2Num; j++) {
                        Reg::LoadUnAlignPre(uReg, diagPtr + i * vlLenC + j * diagStride);
                        Reg::LoadUnAlign(diagReg, uReg, diagPtr + i * vlLenC + j * diagStride);
                        Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, j * offset, mask);
                        if constexpr (sizeof(T) == sizeof(int64_t)) {
                            Reg::UnPack(kMask, tMask);
                            Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                        } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                            Reg::Pack(kMask, tMask);
                            Reg::UnPack((Reg::RegTensor<CastType_>&)diagCastReg, diagReg);
                            Reg::Scatter(xLocalPtr, diagCastReg, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                        } else {
                            Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                        }
                        Reg::Adds(indexReg, indexReg, offset, mask);
                    }
                }
            } else {
                Reg::LoadAlign(indexReg, idxPtr);
                uint32_t temp = oneProcessSize / dataTypeFactor;
                mask = Reg::UpdateMask<MaskType_>(temp);
                for (uint16_t i = 0; i < loop2Num; i++) {
                    Reg::LoadUnAlignPre(uReg, diagPtr + i * diagStride);
                    Reg::LoadUnAlign(diagReg, uReg, diagPtr + i * diagStride);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, i * offset, mask);
                    if constexpr (sizeof(T) == sizeof(int64_t)) {
                        Reg::UnPack(kMask, tMask);
                        Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                        Reg::UnPack((Reg::RegTensor<CastType_>&)diagCastReg, diagReg);
                        Reg::Scatter(xLocalPtr, diagCastReg, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                    } else {
                        Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                    }
                    Reg::Adds(indexReg, indexReg, offset, mask);
                }
                for (uint16_t i = 0; i < hasTail; i++) {
                    uint32_t temp1 = tailSize / dataTypeFactor;
                    mask = Reg::UpdateMask<MaskType_>(temp1);
                    Reg::LoadUnAlignPre(uReg, diagPtr + loop2Num * diagStride);
                    Reg::LoadUnAlign(diagReg, uReg, diagPtr + loop2Num * diagStride);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, loop2Num * offset, mask);
                    if constexpr (sizeof(T) == sizeof(int64_t)) {
                        Reg::UnPack(kMask, tMask);
                        Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                        Reg::UnPack((Reg::RegTensor<CastType_>&)diagCastReg, diagReg);
                        Reg::Scatter(xLocalPtr, diagCastReg, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                    } else {
                        Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void GatherProcess(const LocalTensor<T>& inLocal, const LocalTensor<T>& diagLocal,
                                         const LocalTensor<RangeType_>& idxLocal, uint32_t ProcessNum)
    {
        uint32_t vlLenR = Ops::Base::GetVRegSize() / sizeof(RangeType_);
        uint32_t vlLenC = Ops::Base::GetVRegSize() / sizeof(CastType_);
        uint16_t loop1Num;
        uint16_t loop2Num;
        uint32_t tailSize;
        uint32_t oneProcessSize;
        uint16_t hasTail;

        RangeType_ offset;

        if constexpr (VL_MODE == 0) {
            oneProcessSize = tailAxisDataSize_;
            loop1Num = static_cast<uint16_t>(oneProcessSize) / static_cast<uint16_t>(vlLenC);
            loop2Num = ProcessNum;
            tailSize = (oneProcessSize % vlLenC);
            hasTail = tailSize == 0 ? 0 : 1;
            offset = tailDiagSize_;
        } else {
            loop1Num = static_cast<uint16_t>(vlLenC) / tailAxisDataSize_;
            oneProcessSize = static_cast<uint32_t>(loop1Num) * tailAxisDataSize_;
            loop2Num = static_cast<uint16_t>(ProcessNum) / loop1Num;
            tailSize = (ProcessNum % static_cast<uint32_t>(loop1Num)) * tailAxisDataSize_;
            hasTail = tailSize == 0 ? 0 : 1;
            offset = static_cast<RangeType_>(loop1Num) * static_cast<RangeType_>(tailDiagSize_);
        }
        uint32_t diagStride = oneProcessSize;
        if constexpr (sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int64_t)) {
            oneProcessSize = oneProcessSize * ELEMENT_EXPAND_TIMES;
            tailSize = tailSize * ELEMENT_EXPAND_TIMES;
        }
        uint32_t dataTypeFactor = sizeof(T) > 4 ? 2 : 1;
        auto* xLocalPtr = (__ubuf__ T*)inLocal.GetPhyAddr();
        auto* diagPtr = (__ubuf__ T*)diagLocal.GetPhyAddr();
        auto* idxPtr = (__ubuf__ RangeType_*)idxLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType_> indexReg;
            Reg::RegTensor<T> diagReg;
            Reg::RegTensor<T> inputReg;
            Reg::RegTensor<T> diagCastReg;
            Reg::MaskReg mask;
            Reg::MaskReg tMask;
            Reg::MaskReg kMask;
            Reg::UnalignRegForLoad uReg;

            if constexpr (VL_MODE == 0) {
                for (uint16_t i = 0; i < loop1Num; i++) {
                    mask = Reg::UpdateMask<MaskType_>(oneProcessSize);
                    Reg::LoadAlign(indexReg, idxPtr + i * vlLenR);
                    for (uint16_t j = 0; j < loop2Num; j++) {
                        Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, j * offset, mask);
                        Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, indexReg, (j + 1) * offset, tMask);
                        auto* tempPtr = xLocalPtr + i * vlLenC + j * diagStride;
                        Reg::LoadUnAlignPre(uReg, tempPtr);
                        Reg::LoadUnAlign(inputReg, uReg, tempPtr);
                        if constexpr (sizeof(T) == sizeof(int64_t)) {
                            Reg::UnPack(kMask, tMask);
                            Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                            Reg::Select(diagReg, diagReg, inputReg, kMask);
                            Reg::StoreUnAlign(tempPtr, diagReg, uReg, vlLenC);
                        } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                            Reg::Gather((Reg::RegTensor<CastType_>&)diagReg, diagPtr,
                                        (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                            Reg::Pack(diagCastReg, (Reg::RegTensor<CastType_>&)diagReg);
                            Reg::Pack(kMask, tMask);
                            Reg::Select(diagCastReg, diagCastReg, inputReg, kMask);
                            Reg::StoreUnAlign(tempPtr, diagCastReg, uReg, vlLenC);
                        } else {
                            Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                            Reg::Select(diagReg, diagReg, inputReg, tMask);
                            Reg::StoreUnAlign(tempPtr, diagReg, uReg, vlLenC);
                        }
                        Reg::StoreUnAlignPost(tempPtr, uReg, 0);
                        Reg::Adds(indexReg, indexReg, offset, mask);
                    }
                }
                for (uint16_t i = 0; i < hasTail; i++) {
                    uint32_t temp = tailSize;
                    mask = Reg::UpdateMask<MaskType_>(temp);
                    Reg::LoadAlign(indexReg, idxPtr + loop1Num * vlLenR);
                    for (uint16_t j = 0; j < loop2Num; j++) {
                        Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, j * offset, mask);
                        Reg::Compares<RangeType_, AscendC::CMPMODE::LT>(tMask, indexReg, (j + 1) * offset, tMask);
                        auto* tempPtr = xLocalPtr + loop1Num * vlLenC + j * diagStride;
                        Reg::LoadUnAlignPre(uReg, tempPtr);
                        Reg::LoadUnAlign(inputReg, uReg, tempPtr);
                        if constexpr (sizeof(T) == sizeof(int64_t)) {
                            Reg::UnPack(kMask, tMask);
                            Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                            Reg::Select(diagReg, diagReg, inputReg, kMask);
                            Reg::StoreUnAlign(tempPtr, diagReg, uReg, tailSize / 2);
                        } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                            Reg::Gather((Reg::RegTensor<CastType_>&)diagReg, diagPtr,
                                        (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                            Reg::Pack(diagCastReg, (Reg::RegTensor<CastType_>&)diagReg);
                            Reg::Pack(kMask, tMask);
                            Reg::Select(diagCastReg, diagCastReg, inputReg, kMask);
                            Reg::StoreUnAlign(tempPtr, diagCastReg, uReg, tailSize / 2);
                        } else {
                            Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                            Reg::Select(diagReg, diagReg, inputReg, tMask);
                            Reg::StoreUnAlign(tempPtr, diagReg, uReg, tailSize);
                        }
                        Reg::StoreUnAlignPost(tempPtr, uReg, 0);
                        Reg::Adds(indexReg, indexReg, offset, mask);
                    }
                }
            } else {
                Reg::LoadAlign(indexReg, idxPtr);
                uint32_t temp = oneProcessSize / dataTypeFactor;
                mask = Reg::UpdateMask<MaskType_>(temp);
                for (uint16_t i = 0; i < loop2Num; i++) {
                    auto* tempPtr = xLocalPtr + i * diagStride;
                    Reg::LoadUnAlignPre(uReg, tempPtr);
                    Reg::LoadUnAlign(inputReg, uReg, tempPtr);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, i * offset, mask);
                    if constexpr (sizeof(T) == sizeof(int64_t)) {
                        Reg::UnPack(kMask, tMask);
                        Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                        Reg::Select(diagReg, diagReg, inputReg, kMask);
                        Reg::StoreUnAlign(tempPtr, diagReg, uReg, diagStride);
                    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                        Reg::Gather((Reg::RegTensor<CastType_>&)diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg,
                                    tMask);
                        Reg::Pack(diagCastReg, (Reg::RegTensor<CastType_>&)diagReg);
                        Reg::Pack(kMask, tMask);
                        Reg::Select(diagCastReg, diagCastReg, inputReg, kMask);
                        Reg::StoreUnAlign(tempPtr, diagCastReg, uReg, diagStride);
                    } else {
                        Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                        Reg::Select(diagReg, diagReg, inputReg, tMask);
                        Reg::StoreUnAlign(tempPtr, diagReg, uReg, diagStride);
                    }
                    Reg::StoreUnAlignPost(tempPtr, uReg, 0);
                    Reg::Adds(indexReg, indexReg, offset, mask);
                }
                for (uint16_t i = 0; i < hasTail; i++) {
                    uint32_t temp1 = tailSize / dataTypeFactor;
                    mask = Reg::UpdateMask<MaskType_>(temp1);
                    auto* tempPtr = xLocalPtr + loop2Num * diagStride;
                    Reg::LoadUnAlignPre(uReg, tempPtr);
                    Reg::LoadUnAlign(inputReg, uReg, tempPtr);
                    Reg::Compares<RangeType_, AscendC::CMPMODE::GE>(tMask, indexReg, loop2Num * offset, mask);
                    if constexpr (sizeof(T) == sizeof(int64_t)) {
                        Reg::UnPack(kMask, tMask);
                        Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, kMask);
                        Reg::Select(diagReg, diagReg, inputReg, kMask);
                        Reg::StoreUnAlign(tempPtr, diagReg, uReg, tailSize / 2);
                    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
                        Reg::Gather((Reg::RegTensor<CastType_>&)diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg,
                                    tMask);
                        Reg::Pack(diagCastReg, (Reg::RegTensor<CastType_>&)diagReg);
                        Reg::Pack(kMask, tMask);
                        Reg::Select(diagCastReg, diagCastReg, inputReg, kMask);
                        Reg::StoreUnAlign(tempPtr, diagCastReg, uReg, tailSize / 2);
                    } else {
                        Reg::Gather(diagReg, diagPtr, (Reg::RegTensor<IdxType_>&)indexReg, tMask);
                        Reg::Select(diagReg, diagReg, inputReg, tMask);
                        Reg::StoreUnAlign(tempPtr, diagReg, uReg, tailSize);
                    }
                    Reg::StoreUnAlignPost(tempPtr, uReg, 0);
                }
            }
        }
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetWaitEvent(HardEvent evt, uint32_t bufIdx)
    {
        if (bufIdx & 1) {
            SetFlag<EVENT>(EVENT_ID1);
            WaitFlag<EVENT>(EVENT_ID1);
        } else {
            SetFlag<EVENT>(EVENT_ID0);
            WaitFlag<EVENT>(EVENT_ID0);
        }
    }
};
} // namespace MSD

#endif // ASCENDC_MATRIX_SET_DIAG_NO_CUTW_V2_H_
