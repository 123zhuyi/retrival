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

#ifndef PAD_EDGE_HUGE_WIDTH_H_
#define PAD_EDGE_HUGE_WIDTH_H_

#include "kernel_operator.h"
#include "pad_common.h"
#include "pad_v3_struct.h"

namespace PadV3 {
using namespace AscendC;
struct PadEdgeHugeParam {
    uint32_t padWLOffset;
    uint32_t padWROffset;
};

template <typename T>
class KernelPadEdgeWithHugeWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);

    GlobalTensor<T> input_;
    GlobalTensor<T> output_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
    TPipe* pipe_ = nullptr;

    uint32_t blockIdx_;
    uint32_t additionOffsetHuge_;

    const PadACTilingData* tilingData_ = nullptr;

    uint32_t inCopyLen_;
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

    int8_t ubAxis_;
    uint64_t ubFactor_;

    uint16_t needPadRight_;
    uint32_t padRightLen_;
    T padRightValue_;
    uint16_t repeatTimes_;

    uint64_t factorOfubAxis_;

public:
    __aicore__ inline KernelPadEdgeWithHugeWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();

        additionOffsetHuge_ = tilingData_->additionTileSize / sizeof(T);

        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM, tilingData_->outTileSize + tilingData_->additionTileSize);

        ubAxis_ = tilingData_->ubAxis;
        ubFactor_ = tilingData_->ubFactor;
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdxEdgeHuge = blockIdx_ * tilingData_->ubPerCount;
        if (startIdxEdgeHuge >= tilingData_->ubTotalCount) {
            return;
        }
        uint32_t endIdxEdgeHuge = (blockIdx_ + 1L) * tilingData_->ubPerCount;
        endIdxEdgeHuge = (endIdxEdgeHuge < tilingData_->ubTotalCount ? endIdxEdgeHuge : tilingData_->ubTotalCount);

        factorOfubAxis_ = CeilDiv(tilingData_->outShape[ubAxis_], ubFactor_);

        for (uint32_t idx = startIdxEdgeHuge; idx < endIdxEdgeHuge; idx++) {
            uint32_t curIdx = idx;

            needPadRight_ = 0;
            padRightLen_ = 0;

            for (int32_t i = ubAxis_; i >= 0; i--) {
                uint64_t factor = tilingData_->outShape[i];
                if (i == ubAxis_) {
                    factor = factorOfubAxis_;
                }

                outIndex_[i] = (i == ubAxis_ ? curIdx % factor * ubFactor_ : curIdx % factor);

                if (outIndex_[i] < tilingData_->leftPad[i]) {
                    inIndex_[i] = 0;
                } else {
                    inIndex_[i] = min(outIndex_[i] - tilingData_->leftPad[i], tilingData_->inShape[i] - 1);
                }

                curIdx = curIdx / factor;
            }

            if (outIndex_[ubAxis_] < tilingData_->leftPad[ubAxis_] + tilingData_->inShape[ubAxis_]) {
                if (outIndex_[ubAxis_] + ubFactor_ <= tilingData_->leftPad[ubAxis_]) {
                    inCopyLen_ = 0;
                } else if (outIndex_[ubAxis_] + ubFactor_ <
                           tilingData_->leftPad[ubAxis_] + tilingData_->inShape[ubAxis_]) {
                    inCopyLen_ = outIndex_[ubAxis_] + ubFactor_ - inIndex_[ubAxis_] - tilingData_->leftPad[ubAxis_];
                } else {
                    inCopyLen_ = tilingData_->inShape[ubAxis_] - inIndex_[ubAxis_];
                }
            } else {
                inCopyLen_ = 0;
            }

            ProcessOneStep();
        }
    }

private:
    __aicore__ inline void ProcessOneStep()
    {
        PadEdgeHugeParam padParam;

        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();

        CopyIn(srcLocal, padParam);

        PadOneLine(srcLocal[additionOffsetHuge_], padParam, srcLocal, VL_SIZE / sizeof(T));

        CopyOut(srcLocal, padParam);

        inQueue_.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadEdgeHugeParam& padParam)
    {
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < tilingData_->dimNum; i++) {
            inAddr += inIndex_[i] * tilingData_->inStride[i];
        }

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        if (inCopyLen_ == ubFactor_) {
            padParam.padWLOffset = 0;
            padParam.padWROffset = ubFactor_;
            copyInParams.blockLen = ubFactor_ * sizeof(T);

            DataCopyPad(src[additionOffsetHuge_], input_[inAddr], copyInParams, padParams);
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else if (inCopyLen_ == 0) {
            padParam.padWLOffset = 0;
            padParam.padWROffset = 0;

            SetEvent<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            T padValue = input_[inAddr].GetValue(0);

            Duplicate(src[additionOffsetHuge_], padValue, ubFactor_);
        } else {
            if (outIndex_[ubAxis_] < tilingData_->leftPad[ubAxis_]) {
                padParam.padWLOffset = tilingData_->leftPad[ubAxis_] % ubFactor_;
            } else {
                padParam.padWLOffset = 0;
            }

            if (outIndex_[ubAxis_] + ubFactor_ <= tilingData_->leftPad[ubAxis_] + tilingData_->inShape[ubAxis_]) {
                padParam.padWROffset = ubFactor_;
            } else {
                padParam.padWROffset = (tilingData_->leftPad[ubAxis_] + tilingData_->inShape[ubAxis_]) % ubFactor_;
            }

            uint32_t ubOffset = CeilAlign(padParam.padWLOffset, BLK_ELEMS);

            uint32_t copyLen = inCopyLen_;
            uint32_t remainderLen = 0;

            if (ubOffset != padParam.padWLOffset) {
                remainderLen = ubOffset - padParam.padWLOffset;
                if (inCopyLen_ <= remainderLen) {
                    copyLen = 0;
                    remainderLen = inCopyLen_;
                } else {
                    copyLen = inCopyLen_ - remainderLen;
                }
                copyInParams.blockLen = remainderLen * sizeof(T);
                DataCopyPad(src, input_[inAddr], copyInParams, padParams);
            }

            if (copyLen != 0) {
                copyInParams.blockLen = copyLen * sizeof(T);
                DataCopyPad(src[additionOffsetHuge_ + ubOffset], input_[inAddr + remainderLen], copyInParams,
                            padParams);
            }

            if (padParam.padWLOffset != 0) {
                SetEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                T padValue{0};
                if (ubOffset != padParam.padWLOffset) {
                    padValue = src.GetValue(0);
                } else {
                    padValue = src[additionOffsetHuge_].GetValue(ubOffset);
                }
                Duplicate(src[additionOffsetHuge_], padValue, padParam.padWLOffset);
            }
            if (padParam.padWROffset != ubFactor_) {
                padRightLen_ = min(ubFactor_, tilingData_->outShape[ubAxis_] - outIndex_[ubAxis_]) -
                               padParam.padWROffset;
                if (padRightLen_ != 0) {
                    SetEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    if (copyLen != 0) {
                        padRightValue_ = src[additionOffsetHuge_].GetValue(padParam.padWROffset - 1);
                    } else {
                        padRightValue_ = src.GetValue(inCopyLen_ - 1);
                    }

                    uint32_t additionLen = VL_SIZE / sizeof(T);
                    if (padRightLen_ <= additionLen) {
                        repeatTimes_ = 0;
                    } else {
                        repeatTimes_ = padRightLen_ / additionLen;
                        padRightLen_ -= repeatTimes_ * additionLen;
                    }
                    needPadRight_ = 1;
                }
            }
        }
    }

    __aicore__ inline void PadOneLine(LocalTensor<T> src, PadEdgeHugeParam& padParam, LocalTensor<T> addition,
                                      uint32_t additionLen)
    {
        if (padParam.padWLOffset % BLK_ELEMS == 0 && needPadRight_ == 0) {
            SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            return;
        }
        if constexpr (sizeof(T) == B64_BYTES) {
            PadBothSide<AscendC::Reg::RegTraitNumTwo>(src, padParam, addition, additionLen, needPadRight_, padRightLen_,
                                                      padRightValue_, repeatTimes_);
        } else {
            PadBothSide<AscendC::Reg::RegTraitNumOne>(src, padParam, addition, additionLen, needPadRight_, padRightLen_,
                                                      padRightValue_, repeatTimes_);
        }
        SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadBothSide(LocalTensor<T> dst, PadEdgeHugeParam& padParam, LocalTensor<T> addition,
                                       uint32_t additionLen, uint16_t npr, uint32_t prl, T prv, uint16_t rt)
    {
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* additionAddr = (__ubuf__ T*)addition.GetPhyAddr();

        const uint32_t padWLOffset = padParam.padWLOffset;
        const uint16_t needPadLeft = (padParam.padWLOffset > 0 && padParam.padWLOffset % BLK_ELEMS != 0);

        const uint32_t padLeftCeilAlign = CeilAlign(padParam.padWLOffset, BLK_ELEMS);
        const uint32_t padLeftEdge = padParam.padWROffset > padLeftCeilAlign ? padLeftCeilAlign : padParam.padWROffset;
        const uint32_t padLeftSize = padLeftEdge - padParam.padWLOffset;

        T padRightValue = prv;
        const uint16_t needPadRight = npr;
        const uint16_t needPadRightSurplus = (prl != 0);
        const uint32_t padRightLen = prl;
        const uint16_t repeatTimes = rt;
        const uint32_t padWROffset = padParam.padWROffset;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T, Trait> vReg;
            AscendC::Reg::UnalignReg uReg;

            for (uint16_t k = 0; k < needPadLeft; k++) {
                __ubuf__ T* outAddr = dstAddr + padWLOffset;
                AscendC::Reg::DataCopy(vReg, additionAddr);
                AscendC::Reg::DataCopyUnAlign(outAddr, vReg, uReg, padLeftSize);
                AscendC::Reg::DataCopyUnAlignPost(outAddr, uReg, 0);
            }

            for (uint16_t j = 0; j < needPadRight; j++) {
                __ubuf__ T* outAddr = dstAddr + padWROffset;
                AscendC::Reg::Duplicate(vReg, padRightValue);

                for (uint16_t k = 0; k < repeatTimes; k++) {
                    AscendC::Reg::DataCopyUnAlign(outAddr, vReg, uReg, additionLen);
                }

                for (uint16_t k = 0; k < needPadRightSurplus; k++) {
                    outAddr = dstAddr + padWROffset + repeatTimes * additionLen;
                    AscendC::Reg::DataCopyUnAlign(outAddr, vReg, uReg, padRightLen);
                }
                AscendC::Reg::DataCopyUnAlignPost(outAddr, uReg, 0);
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, PadEdgeHugeParam& padParam)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < tilingData_->dimNum; i++) {
            outAddr += outIndex_[i] * tilingData_->outStride[i];
        }
        uint32_t copyLen = (outIndex_[ubAxis_] + ubFactor_ < tilingData_->outShape[ubAxis_] ?
                                ubFactor_ :
                                tilingData_->outShape[ubAxis_] - outIndex_[ubAxis_]);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(output_[outAddr], src[additionOffsetHuge_], copyOutParams);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventIdEdgeHuge = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventIdEdgeHuge);
        WaitFlag<EVENT>(eventIdEdgeHuge);
    }
};
} // namespace PadV3
#endif
