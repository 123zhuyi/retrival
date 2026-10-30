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

#ifndef PAD_HUGE_WIDTH_H_
#define PAD_HUGE_WIDTH_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;
struct PadHugeParam {
    uint32_t padH;
    uint32_t padW;
    uint32_t padHLOffset;
    uint32_t padHROffset;
    uint32_t padWLOffset;
    uint32_t padWROffset;
};

template <typename T, typename U = int8_t>
class KernelPadWithHugeWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<T> constPad_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;

    TPipe* pipe_ = nullptr;
    int64_t blockIdx_;
    uint32_t additionOffset_{0};
    T constValue_{0};
    const PadACTilingData* tilingData_ = nullptr;
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t inCopyLen_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

public:
    __aicore__ inline KernelPadWithHugeWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR constValue = nullptr)
    {
        blockIdx_ = GetBlockIdx();
        additionOffset_ = tilingData_->additionTileSize / sizeof(T);
        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);
        if (constValue != nullptr) {
            constPad_.SetGlobalBuffer((__gm__ T*)constValue);
            constValue_ = constPad_(0);
            if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
                T padValue = constPad_(0);
                T low4bit = padValue & 0x0F;
                constValue_ = (low4bit << 4) | low4bit;
            }
        }

        pipe_->InitBuffer(inQueue_, BUFFER_NUM, tilingData_->outTileSize + tilingData_->additionTileSize);
        for (uint32_t i = tilingData_->ubAxis + 1U; i < tilingData_->dimNum; i++) {
            inCopyLen_[i] = tilingData_->inShape[i];
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdxHuge = blockIdx_ * tilingData_->ubPerCount;
        if (startIdxHuge >= tilingData_->ubTotalCount) {
            return;
        }
        uint32_t endIdxHuge = (blockIdx_ + 1L) * tilingData_->ubPerCount;
        endIdxHuge = (endIdxHuge < tilingData_->ubTotalCount ? endIdxHuge : tilingData_->ubTotalCount);

        uint8_t ubAxis = tilingData_->ubAxis;
        uint64_t ubFactor = tilingData_->ubFactor;
        for (uint32_t idx = startIdxHuge; idx < endIdxHuge; idx++) {
            bool isAllPadding = false;
            uint32_t curIdxHuge = idx;
            for (int32_t i = ubAxis; i >= 0; i--) {
                uint64_t factorHuge = tilingData_->outShape[i];
                if (i == ubAxis) {
                    factorHuge = CeilDiv(tilingData_->outShape[i], ubFactor);
                }
                if (factorHuge != 0) {
                    outIndex_[i] = (i == ubAxis ? curIdxHuge % factorHuge * ubFactor : curIdxHuge % factorHuge);
                }
                inIndex_[i] = outIndex_[i] < tilingData_->leftPad[i] ? 0 : outIndex_[i] - tilingData_->leftPad[i];
                if (i != ubAxis && (outIndex_[i] < tilingData_->leftPad[i] ||
                                    outIndex_[i] >= tilingData_->inShape[i] + tilingData_->leftPad[i])) {
                    isAllPadding = true;
                }
                if (factorHuge != 0) {
                    curIdxHuge = curIdxHuge / factorHuge;
                }
            }
            if (isAllPadding) {
                inCopyLen_[ubAxis] = 0;
            } else {
                if (outIndex_[ubAxis] < tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis]) {
                    // outIndex 在右pad点的左侧
                    if (outIndex_[ubAxis] + ubFactor <= tilingData_->leftPad[ubAxis]) {
                        // 输出都在左pad点左侧
                        inCopyLen_[ubAxis] = 0;
                    } else if (outIndex_[ubAxis] + ubFactor <
                               tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis]) {
                        // 输出都在右pad点左侧
                        inCopyLen_[ubAxis] = outIndex_[ubAxis] + ubFactor - inIndex_[ubAxis] -
                                             tilingData_->leftPad[ubAxis];
                    } else {
                        // 输出跨过右pad点
                        inCopyLen_[ubAxis] = tilingData_->inShape[ubAxis] - inIndex_[ubAxis];
                    }
                } else {
                    // outIndex 在右pad点的右侧
                    inCopyLen_[ubAxis] = 0;
                }
            }
            ProcessOneStep();
        }
    }

private:
    __aicore__ inline void ProcessOneStep()
    {
        // Copy IN
        LocalTensor<T> srcLocal = inQueue_.AllocTensor<T>();
        PadHugeParam padParam;
        padParam.padH = 1;
        padParam.padW = CeilAlign(tilingData_->ubFactor, BLK_ELEMS);
        padParam.padHLOffset = 0;
        padParam.padHROffset = 1;

        CopyIn(srcLocal, padParam);

        PadOneLine(srcLocal[additionOffset_], padParam, srcLocal, VL_SIZE / sizeof(T));

        CopyOut(srcLocal, padParam);

        inQueue_.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadHugeParam& padParam)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < tilingData_->dimNum; i++) {
            inAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        if (inCopyLen_[ubAxis] == ubFactor) {
            padParam.padWLOffset = 0;
            padParam.padWROffset = ubFactor;
            copyInParams.blockLen = ubFactor * sizeof(T);
            DataCopyPad(src[additionOffset_], input_[inAddr], copyInParams, padParams);
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        } else if (inCopyLen_[ubAxis] == 0) {
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            padParam.padWLOffset = 0;
            padParam.padWROffset = 0;
            Duplicate(src[additionOffset_], constValue_, tilingData_->outTileSize / sizeof(T));
        } else {
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            padParam.padWLOffset = (outIndex_[ubAxis] < tilingData_->leftPad[ubAxis]) ?
                                       tilingData_->leftPad[ubAxis] % ubFactor :
                                       0;
            padParam.padWROffset = (outIndex_[ubAxis] + ubFactor <=
                                    tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis]) ?
                                       ubFactor :
                                       (tilingData_->leftPad[tilingData_->ubAxis] +
                                        tilingData_->inShape[tilingData_->ubAxis]) %
                                           ubFactor;
            uint32_t ubOffset = CeilAlign(padParam.padWLOffset, BLK_ELEMS);
            Duplicate(src, constValue_, additionOffset_ + tilingData_->outTileSize / sizeof(T));
            SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            uint32_t copyLen = inCopyLen_[ubAxis];
            uint32_t remainderLen = 0;
            if (ubOffset != padParam.padWLOffset) {
                remainderLen = ubOffset - padParam.padWLOffset;
                if (inCopyLen_[ubAxis] <= remainderLen) {
                    copyLen = 0;
                    remainderLen = inCopyLen_[ubAxis];
                } else {
                    copyLen = inCopyLen_[ubAxis] - remainderLen;
                }
                copyInParams.blockLen = remainderLen * sizeof(T);
                DataCopyPad(src, input_[inAddr], copyInParams, padParams);
            }
            if (copyLen != 0) {
                copyInParams.blockLen = copyLen * sizeof(T);
                DataCopyPad(src[additionOffset_ + ubOffset], input_[inAddr + remainderLen], copyInParams, padParams);
            }

            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, PadHugeParam& padParam)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < tilingData_->dimNum; i++) {
            outAddr += outIndex_[i] * tilingData_->outStride[i];
        }
        uint32_t copyLen = (outIndex_[ubAxis] + ubFactor < tilingData_->outShape[ubAxis] ?
                                ubFactor :
                                tilingData_->outShape[ubAxis] - outIndex_[ubAxis]);
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        DataCopyPad(output_[outAddr], src[additionOffset_], copyOutParams);
    }

    __aicore__ inline void PadOneLine(LocalTensor<T> src, PadHugeParam& padParam, LocalTensor<T> addition,
                                      uint32_t additionLen)
    {
        if (padParam.padW % BLK_ELEMS == 0 && padParam.padWLOffset % BLK_ELEMS == 0 &&
            padParam.padWROffset % BLK_ELEMS == 0) {
            SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            return;
        }
        if constexpr (sizeof(T) == B64_BYTES) {
            PadBothSide<AscendC::Reg::RegTraitNumTwo>(src, padParam, addition, additionLen);
        } else {
            PadBothSide<AscendC::Reg::RegTraitNumOne>(src, padParam, addition, additionLen);
        }
        SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadBothSide(LocalTensor<T> dst, PadHugeParam& padParam, LocalTensor<T> addition,
                                       uint32_t additionLen)
    {
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* additionAddr = (__ubuf__ T*)addition.GetPhyAddr();

        const T value = constValue_;
        const uint16_t totalHNum = padParam.padHROffset - padParam.padHLOffset;
        const uint32_t dstOffset = padParam.padHLOffset * padParam.padW;
        const uint32_t padW = padParam.padW;
        const uint32_t padWLOffset = padParam.padWLOffset;
        const uint16_t needPadLeft = (padParam.padWLOffset > 0 && padParam.padWLOffset % BLK_ELEMS != 0);
        const uint32_t padLeftCeilAlign = CeilAlign(padParam.padWLOffset, BLK_ELEMS);
        // pad一行的边界，最大到pad的右偏移点
        const uint32_t padLeftEdge = padParam.padWROffset > padLeftCeilAlign ? padLeftCeilAlign : padParam.padWROffset;
        const uint32_t padLeftSize = padLeftEdge - padParam.padWLOffset;
        const uint16_t needPadRight = (padParam.padWROffset > padLeftCeilAlign &&
                                       padParam.padWROffset % BLK_ELEMS != 0);
        const uint32_t padRigthFloorAlign = padParam.padWROffset / BLK_ELEMS * BLK_ELEMS;
        const uint32_t noPadRightSize = padParam.padWROffset - padRigthFloorAlign; // 实际输入右边界

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T, Trait> vReg;
            AscendC::Reg::RegTensor<T, Trait> vRegTmp;
            AscendC::Reg::UnalignReg uReg;
            AscendC::Reg::MaskReg pMask;
            AscendC::Reg::MaskReg outMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, Trait>();
            uint32_t padLen = padLeftSize;

            for (uint16_t k = 0; k < needPadLeft; k++) {
                for (uint16_t n = 0; n < totalHNum; n++) {
                    __ubuf__ T* outAddr = dstAddr + dstOffset + padWLOffset + n * padW;
                    AscendC::Reg::DataCopy(vReg, additionAddr + n * additionLen);
                    AscendC::Reg::DataCopyUnAlign(outAddr, vReg, uReg, padLen);
                    AscendC::Reg::DataCopyUnAlignPost(outAddr, uReg, 0);
                }
            }

            for (uint16_t k = 0; k < needPadRight; k++) {
                uint32_t noPadLen = noPadRightSize;
                uint32_t outLen = BLK_ELEMS;
                pMask = AscendC::Reg::UpdateMask<T, Trait>(noPadLen);
                AscendC::Reg::MaskNot(pMask, pMask, maskAll);
                outMask = AscendC::Reg::UpdateMask<T, Trait>(outLen);
                for (uint16_t n = 0; n < totalHNum; n++) {
                    AscendC::Reg::DataCopy(vReg, dstAddr + dstOffset + padRigthFloorAlign + n * padW);
                    vRegTmp = vReg;
                    Duplicate<T, AscendC::Reg::MaskMergeMode::ZEROING, T>(vRegTmp, value, pMask);
                    Copy(vReg, vRegTmp, pMask);
                    AscendC::Reg::DataCopy(dstAddr + dstOffset + padRigthFloorAlign + n * padW, vReg, outMask);
                }
            }
        }
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventIdHuge = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventIdHuge);
        WaitFlag<EVENT>(eventIdHuge);
    }
};
} // namespace PadV3
#endif
