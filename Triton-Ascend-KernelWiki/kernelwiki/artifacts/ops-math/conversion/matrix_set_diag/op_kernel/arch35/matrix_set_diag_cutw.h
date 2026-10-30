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
 * \file matrix_set_diag_cutw.h
 * \brief matrix_set_diag cutw scatter kernel
 */

#ifndef MATRIX_SET_DIAG_CUTW_H
#define MATRIX_SET_DIAG_CUTW_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "matrix_set_diag_tilingdata.h"

namespace MSD {
using namespace AscendC;

template <typename T>
class MatrixSetDiagCutWScatter {
private:
    constexpr static uint32_t ALIGN_NUM = 32 / sizeof(T);
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static uint32_t NUM_2 = 2;
    constexpr static uint32_t BUFIDX_BIT0 = 1; // 区分奇偶性，用于double buffer同步

private:
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using MaskType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), T, int32_t>;
    using CastType_ = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

    TPipe* pipe_ = nullptr;
    const MatrixSetDiagTilingData* tdPtr_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> diagGm_;
    GlobalTensor<T> outputGm_;

    TBuf<TPosition::VECCALC> inQue_;
    TBuf<TPosition::VECCALC> diagQue_;

    uint32_t blockIdx_{0};
    uint32_t coreNum_{0};
    uint64_t mergeDimSize_{0}; // 非尾轴合轴后的大小
    uint64_t xRowNum_{0};      // x的行数 Dn-2
    uint64_t xColNum_{0};      // x的列数 Dn-1
    uint64_t diagLen_{0};      // 对角线长度，min(Dn-2, Dn-1)
    uint64_t diagLoadLenAlign_{0};
    uint64_t ubPerCore_{0}; // 单核处理的ubFactor个数
    uint64_t ubPerTail_{0}; // 每个尾轴需要切的ubFactor个数
    uint64_t ubFactor_{0};  // 单核每次处理的尾轴数据个数
    uint64_t ubFactorAlign_{0};

    uint16_t vlLen_ = Ops::Base::GetVRegSize() / sizeof(T);

public:
    __aicore__ inline MatrixSetDiagCutWScatter(TPipe* pipe) { pipe_ = pipe; }

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
        ubPerTail_ = tdPtr_->ubPerTail;
        ubFactor_ = tdPtr_->ubFactor;

        diagLoadLenAlign_ = Ops::Base::CeilAlign(ubFactor_ / (xColNum_ + 1) + 1, static_cast<uint64_t>(ALIGN_NUM));
        ubFactorAlign_ = Ops::Base::CeilAlign(ubFactor_, static_cast<uint64_t>(ALIGN_NUM));

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        diagGm_.SetGlobalBuffer((__gm__ T*)diagonal);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQue_, BUF_NUM * ubFactorAlign_ * sizeof(T));
        pipe_->InitBuffer(diagQue_, BUF_NUM * diagLoadLenAlign_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint64_t xOffsetInTail = 0;     // 在尾轴的偏移
        uint64_t xOffsetCurCore = 0;    // x起始地址
        uint32_t xNum = 0;              // 需要搬运的x个数
        uint64_t diagOffsetCurCore = 0; // diag起始地址
        uint32_t diagNum = 0;           // 需要搬运的diag个数
        uint64_t rowIdxInTail = 0;      // 在尾轴范围的行索引
        uint64_t colIdxInTail = 0;      // 在尾轴范围的列索引

        uint32_t bufIdx = 0;
        LocalTensor<T> xLocal = inQue_.Get<T>();
        LocalTensor<T> diagLocal = diagQue_.Get<T>();

        for (uint64_t idx = blockIdx_; idx < ubPerTail_ * mergeDimSize_; idx += coreNum_) {
            LocalTensor<T> x = xLocal[(bufIdx & BUFIDX_BIT0) * ubFactorAlign_];
            LocalTensor<T> diag = diagLocal[(bufIdx & BUFIDX_BIT0) * diagLoadLenAlign_];

            xOffsetInTail = (idx % ubPerTail_) * ubFactor_;
            xOffsetCurCore = (idx / ubPerTail_) * xRowNum_ * xColNum_ + xOffsetInTail;
            xNum = static_cast<uint32_t>(min(xOffsetCurCore + ubFactor_, (idx / ubPerTail_ + 1) * xRowNum_ * xColNum_) -
                                         xOffsetCurCore);

            // x终点地址覆盖的对角线个数 - x起始地址覆盖的对角线个数
            diagNum = min(diagLen_, Ops::Base::CeilDiv(xOffsetInTail + xNum, xColNum_ + 1)) -
                      min(diagLen_, Ops::Base::CeilDiv(xOffsetInTail, xColNum_ + 1));
            diagOffsetCurCore = (idx / ubPerTail_) * diagLen_ + Ops::Base::CeilDiv(xOffsetInTail, xColNum_ + 1);

            rowIdxInTail = xOffsetInTail / xColNum_;
            colIdxInTail = xOffsetInTail % xColNum_;

            if (bufIdx > 1) {
                if (bufIdx & BUFIDX_BIT0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                } else {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
            CopyIn(x, xOffsetCurCore, xNum, diag, diagOffsetCurCore, diagNum);
            SetWaitEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V, bufIdx);
            ScatterAndCopyOut(x, diag, diagNum, rowIdxInTail, colIdxInTail, xOffsetCurCore, xNum, xOffsetInTail,
                              bufIdx);
            bufIdx++;
        }
    }

private:
    __aicore__ inline void VFProcess(__local_mem__ T* diagPtr, __ubuf__ T* xLocalPtr, uint32_t diagNum,
                                     uint16_t loopNum, uint16_t processNum, IdxType_ indexStart)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> diagReg;
            Reg::RegTensor<T> diagCastReg;
            Reg::RegTensor<RangeType_> indexReg;
            Reg::MaskReg mask;
            for (uint16_t i = 0; i < loopNum; i++) {
                mask = Reg::UpdateMask<MaskType_>(diagNum);
                Reg::LoadAlign(diagReg, diagPtr + i * processNum);

                Reg::Arange(indexReg, i * processNum);
                Reg::Muls(indexReg, indexReg, xColNum_ + 1, mask);
                Reg::Adds(indexReg, indexReg, indexStart, mask);
                if constexpr (sizeof(T) != sizeof(int8_t)) {
                    Reg::Scatter(xLocalPtr, diagReg, (Reg::RegTensor<IdxType_>&)indexReg, mask);
                } else {
                    Reg::UnPack((Reg::RegTensor<CastType_>&)diagCastReg, diagReg);
                    Reg::Scatter(xLocalPtr, diagCastReg, (Reg::RegTensor<IdxType_>&)indexReg, mask);
                }
            }
        }
    }
    __aicore__ inline void ScatterAndCopyOut(const LocalTensor<T>& x, const LocalTensor<T>& diag, uint32_t diagNum,
                                             uint64_t rowIdxInTail, uint64_t colIdxInTail, const uint64_t yAddr,
                                             uint32_t yNum, const uint64_t offsetInTail, uint32_t bufIdx)
    {
        // 构建index
        uint64_t indexStart = 0;
        if (rowIdxInTail >= colIdxInTail) {
            // 在下三角或对角线上
            indexStart = rowIdxInTail * xColNum_ + rowIdxInTail;
        } else {
            // 在上三角
            indexStart = (rowIdxInTail + 1) * xColNum_ + rowIdxInTail + 1;
        }
        uint16_t vlLen = vlLen_;
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            vlLen = vlLen / NUM_2;
        }
        uint16_t loopNum = static_cast<uint16_t>(Ops::Base::CeilDiv(diagNum, static_cast<uint32_t>(vlLen)));

        if constexpr (sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int64_t)) {
            diagNum = diagNum * NUM_2;
        }

        auto* diagPtr = (__local_mem__ T*)diag.GetPhyAddr();
        auto* xLocalPtr = (__local_mem__ T*)x.GetPhyAddr();
        VFProcess(diagPtr, xLocalPtr, diagNum, loopNum, vlLen, static_cast<IdxType_>(indexStart - offsetInTail));

        SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3, bufIdx);
        DataCopyExtParams yCopyInParams = {1u, static_cast<uint32_t>(yNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[yAddr], x, yCopyInParams);

        if (bufIdx & BUFIDX_BIT0) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        } else {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& x, const uint64_t xAddr, uint32_t xNum,
                                  const LocalTensor<T>& diag, const uint64_t diagAddr, uint32_t diagNum)
    {
        DataCopyPadExtParams<T> xPadParams{false, 0, static_cast<uint8_t>(Ops::Base::CeilAlign(xNum, ALIGN_NUM) - xNum),
                                           0};
        DataCopyExtParams xCopyInParams = {1u, static_cast<uint32_t>(xNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(x, inputGm_[xAddr], xCopyInParams, xPadParams);

        DataCopyPadExtParams<T> diagPadParams{
            false, 0, static_cast<uint8_t>(Ops::Base::CeilAlign(diagNum, ALIGN_NUM) - diagNum), 0};
        DataCopyExtParams diagCopyInParams = {1u, static_cast<uint32_t>(diagNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(diag, diagGm_[diagAddr], diagCopyInParams, diagPadParams);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetWaitEvent(HardEvent evt, uint32_t bufIdx)
    {
        if (bufIdx & BUFIDX_BIT0) {
            SetFlag<EVENT>(EVENT_ID1);
            WaitFlag<EVENT>(EVENT_ID1);
        } else {
            SetFlag<EVENT>(EVENT_ID0);
            WaitFlag<EVENT>(EVENT_ID0);
        }
    }
};
} // namespace MSD

#endif
