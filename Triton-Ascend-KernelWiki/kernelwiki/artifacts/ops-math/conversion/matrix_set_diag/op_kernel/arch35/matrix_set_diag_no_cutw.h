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
 * \file matrix_set_diag_no_cutw.h
 * \brief
 */

#ifndef ASCENDC_MATRIX_SET_DIAG_NO_CUTW_H_
#define ASCENDC_MATRIX_SET_DIAG_NO_CUTW_H_

#include "kernel_operator.h"
#include "matrix_set_diag_tilingdata.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MSD {
using namespace AscendC;

template <typename T>
class MatrixSetDiagNoCutWScatter {
private:
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static uint32_t ALIGN_NUM = 32 / sizeof(T);
    constexpr static uint16_t NUM_2 = 2;

private:
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using MaskType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), T, int32_t>;
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType_ = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    TPipe* pipe_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> diagonalGm_;
    GlobalTensor<T> outputGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECIN, 1> diagQue_;

    int32_t vlLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    int32_t blockIdx_{0};

    // tiling params
    const MatrixSetDiagTilingData* tdPtr_ = nullptr;
    uint32_t coreNum_{0};
    uint64_t mergeDimSize_{0};
    uint64_t xRowNum_{0};
    uint64_t xColNum_{0};
    uint64_t diagLen_{0};
    uint64_t ubPerCore_{0};
    uint64_t tailAxisDataSize_{0};
    uint64_t ubFactor_{0};

public:
    __aicore__ inline MatrixSetDiagNoCutWScatter(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, const MatrixSetDiagTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tdPtr_ = tilingData;
        coreNum_ = tdPtr_->coreNum;
        mergeDimSize_ = tdPtr_->mergeDimSize;
        xRowNum_ = tdPtr_->xRowNum;
        xColNum_ = tdPtr_->xColNum;
        diagLen_ = tdPtr_->diagLen;
        ubPerCore_ = tdPtr_->ubPerCore;
        tailAxisDataSize_ = tdPtr_->tailAxisDataSize;
        ubFactor_ = tdPtr_->ubFactor;

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        diagonalGm_.SetGlobalBuffer((__gm__ T*)diagonal);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(
            inQue_, BUF_NUM,
            Ops::Base::CeilAlign(ubFactor_ * tailAxisDataSize_, static_cast<uint64_t>(ALIGN_NUM)) * sizeof(T));
        pipe_->InitBuffer(diagQue_, BUF_NUM,
                          Ops::Base::CeilAlign(ubFactor_ * diagLen_, static_cast<uint64_t>(ALIGN_NUM)) * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 计算当前块的起始索引
        uint32_t startIdx = blockIdx_;
        // 计算当前块的结束索引
        uint32_t endIdx = Ceil(mergeDimSize_, ubFactor_);

        uint64_t curMergeDimIdx = 0;
        uint64_t curMergeDimIdxEnd = 0;
        uint32_t copyInNum = 0;
        uint32_t diagNum = 0;

        for (uint32_t idx = startIdx; idx < endIdx; idx += coreNum_) {
            curMergeDimIdx = idx * ubFactor_;
            curMergeDimIdxEnd = min(curMergeDimIdx + ubFactor_, mergeDimSize_);
            copyInNum = static_cast<uint32_t>(curMergeDimIdxEnd - curMergeDimIdx) * tailAxisDataSize_;
            diagNum = static_cast<uint32_t>(diagLen_) * (curMergeDimIdxEnd - curMergeDimIdx);
            CopyIn(curMergeDimIdx * tailAxisDataSize_, curMergeDimIdx * diagLen_, copyInNum, diagNum);
            SetWaitEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            ScatterAndCopyOut(curMergeDimIdx * tailAxisDataSize_, diagNum, copyInNum);
        }
    }

    __aicore__ inline void CopyIn(const uint64_t inputAddr, const uint64_t diagAddr, const uint32_t inputProcessNum,
                                  const uint32_t diagProcessNum)
    {
        LocalTensor<T> inLocal = inQue_.AllocTensor<T>();
        LocalTensor<T> diagLocal = diagQue_.AllocTensor<T>();

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

    __aicore__ inline void ScatterAndCopyOut(const uint64_t outAddr, uint32_t diagProcessNum,
                                             const uint32_t outProcessNum)
    {
        uint32_t vlLen = vlLen_;
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            vlLen = vlLen / NUM_2;
        }
        uint16_t loopNum = Ops::Base::CeilDiv(diagProcessNum, static_cast<uint32_t>(vlLen));
        if constexpr (sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int64_t)) {
            diagProcessNum = diagProcessNum * NUM_2;
        }
        LocalTensor<T> inLocal = inQue_.DeQue<T>();
        LocalTensor<T> diagLocal = diagQue_.DeQue<T>();
        auto* xLocalPtr = (__local_mem__ T*)inLocal.GetPhyAddr();
        auto* diagPtr = (__local_mem__ T*)diagLocal.GetPhyAddr();
        VFProcess(xLocalPtr, diagPtr, diagProcessNum, loopNum, vlLen);
        SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(outProcessNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[outAddr], inLocal, copyOutParams);
        inQue_.FreeTensor(inLocal);
        diagQue_.FreeTensor(diagLocal);
    }

    __aicore__ inline void VFProcess(__local_mem__ T* xLocalPtr, __local_mem__ T* diagPtr, uint32_t diagNum,
                                     uint16_t loopNum, uint32_t vlLen)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> diagReg;
            Reg::RegTensor<RangeType_> indexReg;
            Reg::RegTensor<RangeType_> offsetReg1;
            Reg::RegTensor<RangeType_> offsetReg2;
            Reg::RegTensor<T> diagCastReg;
            Reg::MaskReg mask;
            for (uint16_t i = 0; i < loopNum; i++) {
                mask = Reg::UpdateMask<MaskType_>(diagNum);
                Reg::LoadAlign(diagReg, diagPtr + i * vlLen);
                Reg::Arange(indexReg, static_cast<RangeType_>(i * vlLen));
                Reg::Duplicate(offsetReg1, diagLen_);
                Reg::Div(offsetReg1, indexReg, offsetReg1, mask);
                Reg::Muls(offsetReg2, offsetReg1, diagLen_, mask);
                Reg::Sub(offsetReg2, indexReg, offsetReg2, mask);
                Reg::Muls(indexReg, offsetReg2, xColNum_ + 1, mask);
                Reg::Muls(offsetReg1, offsetReg1, xColNum_ * xRowNum_, mask);
                Reg::Add(indexReg, indexReg, offsetReg1, mask);
                if constexpr (sizeof(T) != sizeof(int8_t)) {
                    Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, mask);
                } else {
                    Reg::UnPack((Reg::RegTensor<CastType_>&)diagCastReg, diagReg);
                    Reg::Scatter(xLocalPtr, diagCastReg, (Reg::RegTensor<IdxType_>&)indexReg, mask);
                }
            }
        }
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetWaitEvent(HardEvent event)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }
};
} // namespace MSD

#endif // ASCENDC_MATRIX_SET_DIAG_NO_CUTW_H_
