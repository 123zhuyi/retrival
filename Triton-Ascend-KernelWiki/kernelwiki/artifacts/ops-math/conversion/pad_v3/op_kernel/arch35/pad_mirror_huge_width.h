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
 * \file pad_cut_last.h
 * \brief pad cut last dim kernel
 */

#ifndef PAD_MIRROR_HUGE_WIDTH_H_
#define PAD_MIRROR_HUGE_WIDTH_H_

#include "kernel_operator.h"
#include "pad_common.h"
#include "pad_v3_struct.h"

namespace PadV3 {
using namespace AscendC;

template <typename T, int32_t KEY>
class KernelPadMirrorWithHugeWidth {
private:
    uint32_t inStart;
    uint32_t outLeftStart;
    uint32_t outRightStart;

    static constexpr uint32_t CONST2 = 2;
    static constexpr uint32_t CONST3 = 3;
    static constexpr uint32_t CONST4 = 4;
    static constexpr uint32_t CONST5 = 5;
    static constexpr int32_t SYMMETRIC_CUT_LAST_DIM_BRANCH = 32010;

    struct outIdxAndTimes {
        uint64_t outIdx[CONST3]{0};
        uint8_t cnt = 1;
    };

    GlobalTensor<T> input_;
    GlobalTensor<T> output_;

    TBuf<TPosition::VECCALC> inQueue_;
    TPipe* pipe_ = nullptr;

    uint32_t blockIdx_;

    const PadACTilingData* tilingData_ = nullptr;

    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t leftUbCopyLen_{0};
    uint32_t rightUbCopyLen_{0};

    uint32_t leftUbStartIdx_{0};
    uint32_t rightUbStartIdx_{0};

    uint64_t outGmLeftIndex_{0};
    uint64_t outGmRightIndex_{0};

    uint8_t mUbAxis{0};
    uint64_t mUbFactor{0};

    uint64_t factorOfmUbAxis{0};
    uint64_t mDataLen{0};

    uint64_t mOutAddr{0};
    uint8_t mDim{0};

    DataCopyExtParams copyOutParams;

    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;

    uint8_t mode{0};

    uint64_t originRightPad{0};
    uint64_t originRightStartIndex{0};

public:
    __aicore__ inline KernelPadMirrorWithHugeWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;

        copyOutParams.blockCount = 1;
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        mode = (KEY == SYMMETRIC_CUT_LAST_DIM_BRANCH);

        inStart = 0;
        outLeftStart = tilingData_->outTileSize / sizeof(T);
        outRightStart = CONST2 * tilingData_->outTileSize / sizeof(T);
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();

        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM * CONST3 * tilingData_->outTileSize);

        mUbFactor = tilingData_->ubFactor;
        mDim = tilingData_->dimNum;
        mUbAxis = tilingData_->ubAxis;
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * tilingData_->ubPerCount;
        if (startIdx >= tilingData_->ubTotalCount) {
            return;
        }

        uint32_t endIdx = (blockIdx_ + 1L) * tilingData_->ubPerCount;
        endIdx = (endIdx < tilingData_->ubTotalCount ? endIdx : tilingData_->ubTotalCount);

        factorOfmUbAxis = CeilDiv(tilingData_->inShape[mUbAxis], mUbFactor);

        originRightPad = tilingData_->outShape[mUbAxis] - tilingData_->leftPad[mUbAxis] - tilingData_->inShape[mUbAxis];
        originRightStartIndex = tilingData_->inShape[mUbAxis] - originRightPad - (!mode);

        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint32_t curIdx = idx;
            uint64_t inAddr = 0;

            for (int32_t i = mUbAxis; i >= 0; i--) {
                uint64_t factor = tilingData_->inShape[i];
                if (i == mUbAxis)
                    factor = factorOfmUbAxis;

                inIndex_[i] = (i == mUbAxis ? curIdx % factor * mUbFactor : curIdx % factor);

                outIndex_[i] = inIndex_[i] + tilingData_->leftPad[i];

                curIdx = curIdx / factor;

                inAddr += inIndex_[i] * tilingData_->inStride[i];
            }

            mDataLen = (inIndex_[mUbAxis] + mUbFactor <= tilingData_->inShape[mUbAxis] ?
                            mUbFactor :
                            tilingData_->inShape[mUbAxis] - inIndex_[mUbAxis]);

            ProcessOneStep(idx - startIdx, inAddr);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(uint32_t idx, uint64_t inAddr)
    {
        LocalTensor<T> srcLocal = inQueue_.Get<T>();
        LocalTensor<T> src = srcLocal[(idx & 1) * 3 * tilingData_->outTileSize / sizeof(T)];

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.blockLen = mDataLen * sizeof(T);

        if (idx > 1) {
            if (idx & 1)
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            else
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        DataCopyPad(src[inStart], input_[inAddr], copyInParams, padParams);

        outIdxAndTimes outIdxCnt[CONST5];
        uint8_t realBase = CONST5 - mDim;

        if (tilingData_->leftPad[mUbAxis] == 0 || inIndex_[mUbAxis] > (tilingData_->leftPad[mUbAxis] - mode)) {
            leftUbCopyLen_ = 0;
            leftUbStartIdx_ = 0;
        } else {
            leftUbStartIdx_ = (!mode && inIndex_[mUbAxis] == 0);
            leftUbCopyLen_ = min(mDataLen, (tilingData_->leftPad[mUbAxis] - mode) - inIndex_[mUbAxis] + 1) -
                             leftUbStartIdx_;
            outIdxCnt[CONST4].outIdx[1] = (tilingData_->leftPad[mUbAxis] - mode) -
                                          (inIndex_[mUbAxis] + leftUbStartIdx_ + leftUbCopyLen_ - 1);
        }

        if (originRightPad == 0 || inIndex_[mUbAxis] + mDataLen <= originRightStartIndex) {
            rightUbCopyLen_ = 0;
            rightUbStartIdx_ = 0;
        } else {
            rightUbStartIdx_ = (originRightStartIndex <= inIndex_[mUbAxis]) ? 0 :
                                                                              originRightStartIndex - inIndex_[mUbAxis];
            rightUbCopyLen_ = mDataLen - (!mode && inIndex_[mUbAxis] + mDataLen == tilingData_->inShape[mUbAxis]) -
                              rightUbStartIdx_;
            outIdxCnt[CONST4].outIdx[CONST2] = CONST2 * (tilingData_->inShape[mUbAxis] - !mode) +
                                               (tilingData_->leftPad[mUbAxis] - mode) -
                                               (inIndex_[mUbAxis] + rightUbStartIdx_ + rightUbCopyLen_ - 1);
        }

        for (uint8_t i = 0; i < mDim - 1; ++i) {
            outIdxCnt[realBase + i].outIdx[0] = outIndex_[i] * tilingData_->outStride[i];
            if (tilingData_->leftPad[i] != 0 && inIndex_[i] >= (!mode) &&
                inIndex_[i] <= (tilingData_->leftPad[i] - mode)) {
                outIdxCnt[realBase + i].outIdx[outIdxCnt[realBase + i].cnt++] = (tilingData_->leftPad[i] - mode -
                                                                                 inIndex_[i]) *
                                                                                tilingData_->outStride[i];
            }

            uint64_t originRightPad = tilingData_->outShape[i] - tilingData_->leftPad[i] - tilingData_->inShape[i];
            if (originRightPad != 0 && inIndex_[i] >= (tilingData_->inShape[i] - originRightPad - (!mode)) &&
                inIndex_[i] < (tilingData_->inShape[i] - (!mode))) {
                outIdxCnt[realBase + i].outIdx[outIdxCnt[realBase + i].cnt++] = (CONST2 *
                                                                                     (tilingData_->inShape[i] - !mode) +
                                                                                 tilingData_->leftPad[i] - mode -
                                                                                 inIndex_[i]) *
                                                                                tilingData_->outStride[i];
            }
        }

        PadOneLine(src, idx);

        CopyOut(src, idx, outIdxCnt);
    }

    __aicore__ inline void PadOneLine(LocalTensor<T> src, uint32_t idx)
    {
        if (leftUbCopyLen_ == 0 && rightUbCopyLen_ == 0) {
            if (idx & 1)
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            else
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

            if (idx & 1)
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID1);
            else
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        } else {
            if (idx & 1)
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            else
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            if (idx & 1)
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            else
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            PadBothSide<AscendC::Reg::RegTraitNumOne>(src);

            if (idx & 1)
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            else
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);

            if (idx & 1)
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            else
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        }
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadBothSide(LocalTensor<T> dst)
    {
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        const uint16_t needPadleftVF = (leftUbCopyLen_ != 0);
        const uint16_t needPadRightVF = (rightUbCopyLen_ != 0);

        uint32_t leftUbCopyLenVF = (uint32_t)leftUbCopyLen_;
        uint32_t rightUbCopyLenVF = (uint32_t)rightUbCopyLen_;

        uint32_t leftUbCopyLenVFB8 = 2 * leftUbCopyLen_;
        uint32_t rightUbCopyLenVFB8 = 2 * rightUbCopyLen_;

        constexpr uint16_t oneRepeatSize = AscendC::GetVecLen() / sizeof(CastType);

        uint16_t repeatLeftTimes = CeilDivision(leftUbCopyLen_, oneRepeatSize);
        uint16_t repeatRightTimes = CeilDivision(rightUbCopyLen_, oneRepeatSize);

        uint16_t rangeStart = GetVecLen() / sizeof(RangeType) - 1;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RangeType> idxReg;
            AscendC::Reg::RegTensor<T> dataReg;
            AscendC::Reg::MaskReg maskReg;

            AscendC::Reg::RegTensor<T> dataB16ToB8Reg;
            AscendC::Reg::MaskReg maskRegLowHalf = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::H>();

            __ubuf__ T* srcAddr = dstAddr + inStart;
            for (uint16_t j = 0; j < needPadleftVF; j++) {
                __ubuf__ T* ubLeftAddr = dstAddr + outLeftStart;
                uint16_t leftIdxStart = leftUbStartIdx_ + leftUbCopyLen_ - 1;
                for (uint16_t k = 0; k < repeatLeftTimes; k++) {
                    maskReg = AscendC::Reg::UpdateMask<CastType>(leftUbCopyLenVF);
                    Reg::Arange<RangeType, AscendC::Reg::IndexOrder::DECREASE_ORDER>(
                        idxReg, (RangeType)(leftIdxStart - rangeStart - k * oneRepeatSize));
                    Reg::DataCopyGather((Reg::RegTensor<CastType>&)dataReg, srcAddr, (Reg::RegTensor<IdxType>&)idxReg,
                                        maskReg);
                    if constexpr (sizeof(T) != 1) {
                        Reg::DataCopy(ubLeftAddr + k * oneRepeatSize, dataReg, maskReg);
                    } else {
                        maskRegLowHalf = AscendC::Reg::UpdateMask<T>(leftUbCopyLenVFB8);
                        Reg::Pack(dataB16ToB8Reg, (Reg::RegTensor<CastType>&)dataReg);
                        Reg::DataCopy(ubLeftAddr + k * oneRepeatSize, dataB16ToB8Reg, maskRegLowHalf);
                    }
                }
            }

            for (uint16_t j = 0; j < needPadRightVF; j++) {
                __ubuf__ T* ubRightAddr = dstAddr + outRightStart;
                uint16_t rightIdxStart = rightUbStartIdx_ + rightUbCopyLen_ - 1;
                for (uint16_t k = 0; k < repeatRightTimes; k++) {
                    maskReg = AscendC::Reg::UpdateMask<CastType>(rightUbCopyLenVF);
                    Reg::Arange<RangeType, AscendC::Reg::IndexOrder::DECREASE_ORDER>(
                        idxReg, (RangeType)(rightIdxStart - rangeStart - k * oneRepeatSize));
                    Reg::DataCopyGather((Reg::RegTensor<CastType>&)dataReg, srcAddr, (Reg::RegTensor<IdxType>&)idxReg,
                                        maskReg);
                    if constexpr (sizeof(T) != 1) {
                        Reg::DataCopy(ubRightAddr + k * oneRepeatSize, dataReg, maskReg);
                    } else {
                        maskRegLowHalf = AscendC::Reg::UpdateMask<T>(rightUbCopyLenVFB8);
                        Reg::Pack(dataB16ToB8Reg, (Reg::RegTensor<CastType>&)dataReg);
                        Reg::DataCopy(ubRightAddr + k * oneRepeatSize, dataB16ToB8Reg, maskRegLowHalf);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutOneLine(const LocalTensor<T>& src, uint64_t outAddr, outIdxAndTimes* outIdxCnt)
    {
        uint64_t tempOutAddr = outAddr + outIndex_[mUbAxis];
        copyOutParams.blockLen = mDataLen * sizeof(T);
        DataCopyPad(output_[tempOutAddr], src[inStart], copyOutParams);

        if (leftUbCopyLen_ != 0) {
            copyOutParams.blockLen = leftUbCopyLen_ * sizeof(T);
            tempOutAddr = outAddr + outIdxCnt[CONST4].outIdx[1];
            DataCopyPad(output_[tempOutAddr], src[outLeftStart], copyOutParams);
        }

        if (rightUbCopyLen_ != 0) {
            copyOutParams.blockLen = rightUbCopyLen_ * sizeof(T);
            tempOutAddr = outAddr + outIdxCnt[CONST4].outIdx[CONST2];
            DataCopyPad(output_[tempOutAddr], src[outRightStart], copyOutParams);
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, uint32_t idx, outIdxAndTimes* outIdxCnt)
    {
        for (uint8_t a0 = 0; a0 < outIdxCnt[0].cnt; ++a0) {
            uint64_t a0Offset = outIdxCnt[0].outIdx[a0];
            for (uint8_t a1 = 0; a1 < outIdxCnt[1].cnt; ++a1) {
                uint64_t a1Offset = a0Offset + outIdxCnt[1].outIdx[a1];
                for (uint8_t a2 = 0; a2 < outIdxCnt[CONST2].cnt; ++a2) {
                    uint64_t a2Offset = a1Offset + outIdxCnt[CONST2].outIdx[a2];
                    for (uint8_t a3 = 0; a3 < outIdxCnt[CONST3].cnt; ++a3) {
                        uint64_t a3Offset = a2Offset + outIdxCnt[3].outIdx[a3];
                        CopyOutOneLine(src, a3Offset, outIdxCnt);
                    }
                }
            }
        }

        if (idx & 1)
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        else
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }
};
} // namespace PadV3
#endif
