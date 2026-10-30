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
 * \file pad_circular_huge_width.h
 * \brief pad circular mode implement for w axis situation
 */

#ifndef PAD_CIRCULAR_HUGE_WIDTH_H_
#define PAD_CIRCULAR_HUGE_WIDTH_H_

#include "kernel_operator.h"
#include "pad_common.h"
#include "pad_v3_struct.h"

namespace PadV3 {
using namespace AscendC;

template <typename T>
class KernelPadCircularWithHugeWidth {
private:
    uint32_t inStart;

    static constexpr uint32_t CONST2 = 2;
    static constexpr uint32_t CONST3 = 3;
    static constexpr uint32_t CONST4 = 4;

    struct outIdxAndTimes {
        uint64_t outGmIdx[CONST3]{0};
        uint8_t cnt = 1;
    };

    GlobalTensor<T> input_;
    GlobalTensor<T> output_;

    TBuf<TPosition::VECCALC> inQueue_;
    TPipe* pipe_ = nullptr;

    uint32_t blockIdx_;

    const PadACTilingData* mTD = nullptr;

    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t leftUbStartIdx_{0};
    uint32_t leftUbCopyLen_{0};
    uint64_t leftOutGmIdx{0};

    uint32_t rightUbCopyLen_{0};
    uint64_t rightOutGmIdx{0};

    uint32_t leftUnalignUbStartIdx_{0};
    uint32_t leftUnalignLen{0};
    uint64_t leftUnalignOutGmIdx{0};

    uint8_t mUbAxis{0};
    uint64_t mUbFactor{0};

    uint64_t factorOfmUbAxis{0};
    uint64_t mDataLen{0};
    uint8_t mDim{0};

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyInParams;

    DataCopyExtParams copyOutParams;

    uint64_t originRightPad{0};

    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);

public:
    __aicore__ inline KernelPadCircularWithHugeWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        mTD = tilingData;

        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        copyOutParams.blockCount = 1;
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();

        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM * (mTD->additionTileSize + mTD->outTileSize));

        mUbFactor = mTD->ubFactor;
        mDim = mTD->dimNum;
        mUbAxis = mTD->ubAxis;

        inStart = mTD->additionTileSize / sizeof(T);
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * mTD->ubPerCount;
        if (startIdx >= mTD->ubTotalCount) {
            return;
        }

        uint32_t endIdx = (blockIdx_ + 1L) * mTD->ubPerCount;
        endIdx = (endIdx < mTD->ubTotalCount ? endIdx : mTD->ubTotalCount);

        factorOfmUbAxis = CeilDiv(mTD->inShape[mUbAxis], mUbFactor);

        originRightPad = mTD->outShape[mUbAxis] - mTD->leftPad[mUbAxis] - mTD->inShape[mUbAxis];

        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint32_t curIdx = idx;
            uint64_t inAddr = 0;

            for (int32_t i = mUbAxis; i >= 0; i--) {
                uint64_t factor = mTD->inShape[i];
                if (i == mUbAxis)
                    factor = factorOfmUbAxis;

                inIndex_[i] = (i == mUbAxis ? curIdx % factor * mUbFactor : curIdx % factor);

                outIndex_[i] = inIndex_[i] + mTD->leftPad[i];

                curIdx = curIdx / factor;

                inAddr += inIndex_[i] * mTD->inStride[i];
            }

            mDataLen = (inIndex_[mUbAxis] + mUbFactor <= mTD->inShape[mUbAxis] ?
                            mUbFactor :
                            mTD->inShape[mUbAxis] - inIndex_[mUbAxis]);

            ProcessOneStep(idx - startIdx, inAddr);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(uint32_t idx, uint64_t inAddr)
    {
        LocalTensor<T> srcLocal = inQueue_.Get<T>();
        LocalTensor<T> src = srcLocal[(idx & 1) * (mTD->additionTileSize + mTD->outTileSize) / sizeof(T)];

        copyInParams.blockLen = mDataLen * sizeof(T);

        if (idx > 1) {
            if (idx & 1)
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            else
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        DataCopyPad(src[inStart], input_[inAddr], copyInParams, padParams);

        outIdxAndTimes outIdxCnt[CONST4];

        uint64_t needLeftStart = mTD->inShape[mUbAxis] - mTD->leftPad[mUbAxis];

        if (mTD->leftPad[mUbAxis] == 0 || inIndex_[mUbAxis] + mDataLen <= needLeftStart) {
            leftUbCopyLen_ = 0;
            leftUbStartIdx_ = 0;
        } else {
            leftUbStartIdx_ = (inIndex_[mUbAxis] < needLeftStart) ? (needLeftStart - inIndex_[mUbAxis]) : 0;
            leftUbCopyLen_ = mDataLen - leftUbStartIdx_;
            leftOutGmIdx = inIndex_[mUbAxis] + leftUbStartIdx_ - needLeftStart;
        }

        if (originRightPad == 0 || inIndex_[mUbAxis] >= originRightPad) {
            rightUbCopyLen_ = 0;
        } else {
            rightUbCopyLen_ = (inIndex_[mUbAxis] + mDataLen <= originRightPad) ? mDataLen :
                                                                                 originRightPad - inIndex_[mUbAxis];
            rightOutGmIdx = mTD->inShape[mUbAxis] + mTD->leftPad[mUbAxis] + inIndex_[mUbAxis];
        }

        for (uint8_t i = 0; i < mDim - 1; ++i) {
            outIdxCnt[i].outGmIdx[0] = outIndex_[i] * mTD->outStride[i];

            if (mTD->leftPad[i] != 0 && inIndex_[i] >= mTD->inShape[i] - mTD->leftPad[i]) {
                outIdxCnt[i].outGmIdx[outIdxCnt[i].cnt++] = (inIndex_[i] - (mTD->inShape[i] - mTD->leftPad[i])) *
                                                            mTD->outStride[i];
            }

            uint64_t originRightPadHigh = mTD->outShape[i] - mTD->leftPad[i] - mTD->inShape[i];
            if (originRightPadHigh != 0 && inIndex_[i] <= (originRightPadHigh - 1)) {
                outIdxCnt[i].outGmIdx[outIdxCnt[i].cnt++] = (mTD->inShape[i] + mTD->leftPad[i] + inIndex_[i]) *
                                                            mTD->outStride[i];
            }
        }

        leftUnalignLen = 0;

        if (leftUbCopyLen_ != 0 && (leftUbStartIdx_ % BLK_ELEMS) != 0) {
            leftUnalignUbStartIdx_ = leftUbStartIdx_;
            leftUnalignLen = CeilAlign(leftUbStartIdx_, BLK_ELEMS) - leftUbStartIdx_;
            leftUnalignOutGmIdx = inIndex_[mUbAxis] + leftUnalignUbStartIdx_ - needLeftStart;

            leftUbStartIdx_ = leftUbStartIdx_ + leftUnalignLen;

            if (leftUnalignLen >= leftUbCopyLen_) {
                leftUnalignLen = leftUbCopyLen_;
                leftUbCopyLen_ = 0;
            } else {
                leftUbCopyLen_ -= leftUnalignLen;
                leftOutGmIdx = inIndex_[mUbAxis] + leftUbStartIdx_ - needLeftStart;
            }
        }

        PadOneLine(src, idx);

        CopyOut(src, idx, outIdxCnt);
    }

    __aicore__ inline void PadOneLine(LocalTensor<T> src, uint32_t idx)
    {
        if (leftUnalignLen == 0) {
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

            PadBothSide(src);

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

    __aicore__ inline void PadBothSide(LocalTensor<T> src)
    {
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* srcLeftDataAddr = srcAddr + inStart + leftUnalignUbStartIdx_;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> vReg;
            AscendC::Reg::UnalignReg uReg;
            AscendC::Reg::UnalignReg uReg1;

            AscendC::Reg::DataCopyUnAlignPre(uReg, srcLeftDataAddr);
            AscendC::Reg::DataCopyUnAlign(vReg, uReg, srcLeftDataAddr);
            AscendC::Reg::DataCopyUnAlign(srcAddr, vReg, uReg1, leftUnalignLen);
            AscendC::Reg::DataCopyUnAlignPost(srcAddr, uReg1, 0);
        }
    }

    __aicore__ inline void CopyOutOneLine(const LocalTensor<T>& src, uint64_t outAddr)
    {
        uint64_t tempOutAddr = outAddr + outIndex_[mUbAxis];
        copyOutParams.blockLen = mDataLen * sizeof(T);
        DataCopyPad(output_[tempOutAddr], src[inStart], copyOutParams);

        if (leftUbCopyLen_ != 0) {
            copyOutParams.blockLen = leftUbCopyLen_ * sizeof(T);
            tempOutAddr = outAddr + leftOutGmIdx;
            DataCopyPad(output_[tempOutAddr], src[inStart + leftUbStartIdx_], copyOutParams);
        }

        if (rightUbCopyLen_ != 0) {
            copyOutParams.blockLen = rightUbCopyLen_ * sizeof(T);
            tempOutAddr = outAddr + rightOutGmIdx;
            DataCopyPad(output_[tempOutAddr], src[inStart], copyOutParams);
        }

        if (leftUnalignLen != 0) {
            copyOutParams.blockLen = leftUnalignLen * sizeof(T);
            tempOutAddr = outAddr + leftUnalignOutGmIdx;
            DataCopyPad(output_[tempOutAddr], src, copyOutParams);
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, uint32_t idx, outIdxAndTimes* outIdxCnt)
    {
        for (uint8_t a0 = 0; a0 < outIdxCnt[0].cnt; ++a0) {
            uint64_t a0Offset = outIdxCnt[0].outGmIdx[a0];
            for (uint8_t a1 = 0; a1 < outIdxCnt[1].cnt; ++a1) {
                uint64_t a1Offset = a0Offset + outIdxCnt[1].outGmIdx[a1];
                for (uint8_t a2 = 0; a2 < outIdxCnt[CONST2].cnt; ++a2) {
                    uint64_t a2Offset = a1Offset + outIdxCnt[CONST2].outGmIdx[a2];
                    for (uint8_t a3 = 0; a3 < outIdxCnt[CONST3].cnt; ++a3) {
                        uint64_t a3Offset = a2Offset + outIdxCnt[CONST3].outGmIdx[a3];
                        CopyOutOneLine(src, a3Offset);
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
