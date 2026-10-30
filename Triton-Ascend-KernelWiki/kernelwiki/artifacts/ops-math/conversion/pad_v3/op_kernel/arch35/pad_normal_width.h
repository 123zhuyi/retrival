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

#ifndef PAD_NORMAL_WIDTH_H_
#define PAD_NORMAL_WIDTH_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;

struct PadNormalParam {
    uint32_t padH;
    uint32_t padW;
    uint32_t padStride[MAX_H_DIMS];
    uint32_t padHLOffset[MAX_H_DIMS];
    uint32_t padHROffset[MAX_H_DIMS];
    uint32_t padWLOffset;
    uint32_t padWROffset;
};

template <typename T, int32_t KEY, typename U = int8_t>
class KernelPadWithNormalWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    static constexpr uint32_t UB_AXES = (KEY / KEY_BASE) % KEY_BASE; // TilingKey倒数第二维为UB内轴个数
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    GlobalTensor<T> constPad_;
    TBuf<TPosition::VECCALC> inQueue_;

    TPipe* pipe_ = nullptr;
    int64_t blockIdx_;
    T constValue_{0};
    uint32_t additionOffset_{0};
    const PadACTilingData* tilingData_ = nullptr;
    bool firstOut_{true};
    bool lastOut_{true};
    uint32_t padHLength_{1};
    uint32_t padWLength_{0};
    uint32_t padStride_[MAX_H_DIMS] = {0, 0, 0};
    uint32_t padHLOffset_[MAX_H_DIMS] = {0, 0, 0};
    uint32_t padHROffset_[MAX_H_DIMS] = {1, 1, 1};
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t inCopyLen_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

public:
    __aicore__ inline KernelPadWithNormalWidth(TPipe* pipe, const PadACTilingData* tilingData)
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

        pipe_->InitBuffer(inQueue_, BUFFER_NUM * (tilingData_->outTileSize + tilingData_->additionTileSize));

        const int8_t dimNum = tilingData_->dimNum;
        const int8_t ubAxis = tilingData_->ubAxis;
        padWLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->outShape[dimNum - 1]), BLK_ELEMS);

        for (int8_t i = dimNum - 1; i > ubAxis; i--) {
            inCopyLen_[i] = tilingData_->inShape[i];
            if constexpr (UB_AXES > 2) {
                // UB内超过两个才需赋值下标1之后的参数
                if (i != dimNum - 1) {
                    padStride_[i - ubAxis] = padHLength_ * padWLength_;
                    padHLength_ = padHLength_ * tilingData_->outShape[i];
                    padHLOffset_[i - ubAxis] = tilingData_->leftPad[i];
                    padHROffset_[i - ubAxis] = tilingData_->inShape[i] + tilingData_->leftPad[i];
                }
            }
        }
        padStride_[0] = padHLength_ * padWLength_;
        padHLength_ = padHLength_ * tilingData_->ubFactor;
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdxNormal = blockIdx_ * tilingData_->ubPerCount;
        if (startIdxNormal >= tilingData_->ubTotalCount) {
            return;
        }
        // 首次的dup在这里做，且设置两个EVENT_ID0
        LocalTensor<T> in = inQueue_.Get<T>();
        Duplicate(in, constValue_, BUFFER_NUM * (tilingData_->outTileSize + tilingData_->additionTileSize) / sizeof(T));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        uint32_t endIdxNormal = (blockIdx_ + 1L) * tilingData_->ubPerCount;
        endIdxNormal = (endIdxNormal < tilingData_->ubTotalCount ? endIdxNormal : tilingData_->ubTotalCount);
        uint8_t ubAxis = tilingData_->ubAxis;
        uint64_t ubFactor = tilingData_->ubFactor;
        for (uint32_t idx = startIdxNormal; idx < endIdxNormal; idx++) {
            bool isAllPadding = false;
            uint32_t curIdx = idx;
            firstOut_ = (idx == startIdxNormal);
            lastOut_ = (idx == endIdxNormal - 1);
            for (int32_t i = ubAxis; i >= 0; i--) {
                uint64_t factorNormal = tilingData_->outShape[i];
                factorNormal = (i == ubAxis) ? CeilDiv(tilingData_->outShape[i], ubFactor) : factorNormal;
                if (factorNormal != 0) {
                    outIndex_[i] = (i == ubAxis ? curIdx % factorNormal * ubFactor : curIdx % factorNormal);
                }
                inIndex_[i] = outIndex_[i] < tilingData_->leftPad[i] ? 0 : outIndex_[i] - tilingData_->leftPad[i];
                if (i != ubAxis && (outIndex_[i] < tilingData_->leftPad[i] ||
                                    outIndex_[i] >= tilingData_->inShape[i] + tilingData_->leftPad[i])) {
                    isAllPadding = true;
                }
                curIdx = (factorNormal != 0) ? curIdx / factorNormal : curIdx;
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
            ProcessOneStep(idx - startIdxNormal);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(int32_t idx)
    {
        // Copy IN
        LocalTensor<T> input = inQueue_.Get<T>();
        LocalTensor<T>
            srcLocal = input[(idx & 1) * (tilingData_->outTileSize + tilingData_->additionTileSize) / sizeof(T)];
        PadNormalParam padParam = {.padH = padHLength_,
                                   .padW = padWLength_,
                                   .padStride = {padStride_[0], padStride_[1], padStride_[2]},
                                   .padHLOffset = {0, padHLOffset_[1], padHLOffset_[2]},
                                   .padHROffset = {tilingData_->ubFactor, padHROffset_[1], padHROffset_[2]},
                                   .padWLOffset = 0,
                                   .padWROffset = static_cast<uint32_t>(tilingData_->inShape[tilingData_->dimNum - 1])};
        bool hasMTE2 = false;
        bool hasPadding = false;
        CopyIn(srcLocal, padParam, hasMTE2, idx);

        PadOneLine(srcLocal, padParam, hasMTE2, hasPadding);

        CopyOut(srcLocal, padParam, hasMTE2, hasPadding, idx);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadNormalParam& padParam, bool& hasMTE2, int32_t idx)
    {
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            LocalTensor<T> input = inQueue_.Get<T>();
            LocalTensor<T> nextLocal = input[(nextIdx & 1) *
                                             (tilingData_->outTileSize + tilingData_->additionTileSize) / sizeof(T)];
            Duplicate(nextLocal, constValue_, (tilingData_->outTileSize + tilingData_->additionTileSize) / sizeof(T));
            if (!lastOut_) {
                if ((nextIdx & 1) == 0) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                } else {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                }
            }
        }

        if ((idx & 1) == 0) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        }
        const uint8_t ubAxis = tilingData_->ubAxis;
        if (inCopyLen_[ubAxis] != 0) {
            DoCopyIn(src, padParam);
            hasMTE2 = true;
        }
    }

    __aicore__ inline void DoCopyIn(const LocalTensor<T>& src, PadNormalParam& padParam)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum; i++) {
            inAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if (inCopyLen_[ubAxis] != ubFactor) {
            padParam.padHLOffset[0] = (outIndex_[ubAxis] < tilingData_->leftPad[ubAxis]) ?
                                          tilingData_->leftPad[ubAxis] % ubFactor :
                                          padParam.padHLOffset[0];
            if (outIndex_[ubAxis] + ubFactor > tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis]) {
                padParam.padHROffset[0] = (tilingData_->leftPad[tilingData_->ubAxis] +
                                           tilingData_->inShape[tilingData_->ubAxis]) %
                                          ubFactor;
            }
        }
        uint32_t ubInOffset = 0;
        for (auto i = 0; i < MAX_H_DIMS; i++) {
            ubInOffset += padParam.padHLOffset[i] * padParam.padStride[i];
        }

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = inCopyLen_[dimNum - DIM_INDEX_SECOND];
        copyInParams.blockLen = tilingData_->inShape[dimNum - 1] * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (padParam.padW - tilingData_->inShape[dimNum - 1]) / BLK_ELEMS;

        if constexpr (UB_AXES == 3) {
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = inCopyLen_[dimNum - DIM_INDEX_THIRD];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum - DIM_INDEX_THIRD] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->outShape[dimNum - DIM_INDEX_SECOND] * padParam.padW * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[additionOffset_ + ubInOffset], input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else if constexpr (UB_AXES == 4) {
            LoopModeParams loopParams;
            loopParams.loop1Size = inCopyLen_[dimNum - DIM_INDEX_THIRD];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum - DIM_INDEX_THIRD] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->outShape[dimNum - DIM_INDEX_SECOND] * padParam.padW * sizeof(T);
            loopParams.loop2Size = inCopyLen_[dimNum - DIM_INDEX_FOURTH];
            loopParams.loop2SrcStride = tilingData_->inStride[dimNum - DIM_INDEX_FOURTH] * sizeof(T);
            loopParams.loop2DstStride = tilingData_->outShape[dimNum - DIM_INDEX_THIRD] *
                                        tilingData_->outShape[dimNum - DIM_INDEX_SECOND] * padParam.padW * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[additionOffset_ + ubInOffset], input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else {
            DataCopyPad(src[additionOffset_ + ubInOffset], input_[inAddr], copyInParams, padParams);
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, PadNormalParam& padParam, bool hasMTE2, bool hasPadding,
                                   int32_t idx)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        const uint32_t ubFactor = tilingData_->ubFactor;
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < dimNum; i++) {
            outAddr += outIndex_[i] * tilingData_->outStride[i];
        }
        uint32_t blockCount = (outIndex_[ubAxis] + ubFactor < tilingData_->outShape[ubAxis] ?
                                   ubFactor :
                                   tilingData_->outShape[ubAxis] - outIndex_[ubAxis]);
        for (auto i = ubAxis + 1; i < dimNum - 1; i++) {
            // blockCount 计算到倒数第二维
            blockCount = blockCount * tilingData_->outShape[i];
        }
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = blockCount;
        copyOutParams.blockLen = tilingData_->outShape[dimNum - 1] * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        if (hasMTE2 && !hasPadding) {
            SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        } else {
            SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }
        if (firstOut_ && tilingData_->leftPad[dimNum - 1] != 0) {
            DataCopyExtParams copyOutParams0;
            copyOutParams0.blockCount = 1;
            copyOutParams0.blockLen = tilingData_->leftPad[dimNum - 1] * sizeof(T);
            DataCopyPad(output_[outAddr], src, copyOutParams0);
        }
        if (lastOut_) {
            // 先copy前blockCount - 1行，最后一行单独拷贝
            copyOutParams.blockCount = blockCount - 1;
            if (copyOutParams.blockCount > 0) {
                DataCopyPad(output_[outAddr + tilingData_->leftPad[dimNum - 1]], src[additionOffset_], copyOutParams);
            }
            DataCopyExtParams copyOutParams1;
            copyOutParams1.blockCount = 1;
            copyOutParams1.blockLen = (tilingData_->outShape[dimNum - 1] - tilingData_->leftPad[dimNum - 1]) *
                                      sizeof(T);
            DataCopyPad(output_[outAddr + tilingData_->leftPad[dimNum - 1] +
                                copyOutParams.blockCount * tilingData_->outShape[dimNum - 1]],
                        src[additionOffset_ + copyOutParams.blockCount * padParam.padW], copyOutParams1);
        } else {
            DataCopyPad(output_[outAddr + tilingData_->leftPad[dimNum - 1]], src[additionOffset_], copyOutParams);
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void PadOneLine(LocalTensor<T> src, PadNormalParam& padParam, bool hasMTE2, bool& hasPadding)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        if (inCopyLen_[ubAxis] == 0) {
            return;
        }
        if (padParam.padW % BLK_ELEMS == 0 && padParam.padWLOffset % BLK_ELEMS == 0 &&
            padParam.padWROffset % BLK_ELEMS == 0) {
            return;
        }
        if (hasMTE2) {
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        }
        if constexpr (sizeof(T) == B64_BYTES) {
            PadRigthSide<AscendC::Reg::RegTraitNumTwo>(src[additionOffset_], padParam);
        } else {
            PadRigthSide<AscendC::Reg::RegTraitNumOne>(src[additionOffset_], padParam);
        }
        hasPadding = true;
    }

    template <const AscendC::Reg::RegTrait& Trait>
    __aicore__ inline void PadRigthSide(const LocalTensor<T>& dst, PadNormalParam& padParam)
    {
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        const T value = constValue_;
        const uint16_t dimNNum = padParam.padHROffset[0] - padParam.padHLOffset[0];
        const uint16_t dimCNum = padParam.padHROffset[1] - padParam.padHLOffset[1];
        const uint16_t dimHNum = padParam.padHROffset[2] - padParam.padHLOffset[2];
        const uint32_t padW = padParam.padStride[2];
        const uint32_t padHW = padParam.padStride[1];
        const uint32_t padCHW = padParam.padStride[0];
        const uint32_t padRigthFloorAlign = padParam.padWROffset / BLK_ELEMS * BLK_ELEMS;
        const uint32_t noPadRightSize = padParam.padWROffset - padRigthFloorAlign; // 实际输入右边界
        uint32_t ubInOffset = 0;
        for (auto i = 0; i < MAX_H_DIMS; i++) {
            ubInOffset += padParam.padHLOffset[i] * padParam.padStride[i];
        }

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T, Trait> vReg;
            AscendC::Reg::RegTensor<T, Trait> vRegTmp;
            AscendC::Reg::MaskReg pMask;
            AscendC::Reg::MaskReg outMask;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL, Trait>();

            uint32_t noPadLen = noPadRightSize;
            uint32_t outLen = BLK_ELEMS;
            pMask = AscendC::Reg::UpdateMask<T, Trait>(noPadLen);
            AscendC::Reg::MaskNot(pMask, pMask, maskAll);
            outMask = AscendC::Reg::UpdateMask<T, Trait>(outLen);
            if constexpr (UB_AXES == 2) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    AscendC::Reg::DataCopy(vReg, dstAddr + ubInOffset + padRigthFloorAlign + n * padCHW);
                    vRegTmp = vReg;
                    Duplicate<T, AscendC::Reg::MaskMergeMode::ZEROING, T>(vRegTmp, value, pMask);
                    Copy(vReg, vRegTmp, pMask);
                    AscendC::Reg::DataCopy(dstAddr + ubInOffset + padRigthFloorAlign + n * padCHW, vReg, outMask);
                }
            } else if constexpr (UB_AXES == 3) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        AscendC::Reg::DataCopy(vReg,
                                               dstAddr + ubInOffset + padRigthFloorAlign + n * padCHW + c * padHW);
                        vRegTmp = vReg;
                        Duplicate<T, AscendC::Reg::MaskMergeMode::ZEROING, T>(vRegTmp, value, pMask);
                        Copy(vReg, vRegTmp, pMask);
                        AscendC::Reg::DataCopy(dstAddr + ubInOffset + padRigthFloorAlign + n * padCHW + c * padHW, vReg,
                                               outMask);
                    }
                }
            } else {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        for (uint16_t h = 0; h < dimHNum; h++) {
                            AscendC::Reg::DataCopy(
                                vReg, dstAddr + ubInOffset + padRigthFloorAlign + n * padCHW + c * padHW + h * padW);
                            vRegTmp = vReg;
                            Duplicate<T, AscendC::Reg::MaskMergeMode::ZEROING, T>(vRegTmp, value, pMask);
                            Copy(vReg, vRegTmp, pMask);
                            AscendC::Reg::DataCopy(
                                dstAddr + ubInOffset + padRigthFloorAlign + n * padCHW + c * padHW + h * padW, vReg,
                                outMask);
                        }
                    }
                }
            }
        }
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
