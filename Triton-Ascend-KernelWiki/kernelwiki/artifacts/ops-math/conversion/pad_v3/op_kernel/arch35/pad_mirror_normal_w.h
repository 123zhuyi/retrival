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
 * \file pad_repl_normal_w.h
 * \brief pad cut not last dim kernel in replicate mode
 */

#ifndef PAD_MIRR_NORMAL_W_H_
#define PAD_MIRR_NORMAL_W_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;
struct PadMirrNormalParam {
    uint32_t padH;
    uint32_t padWI;
    uint32_t padWO;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t padStride[MAX_H_DIMS];
};
template <typename T, int32_t KEY>
class KernelPadMirrWithNormalWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    static constexpr uint32_t VL_ELEMS = VL_SIZE / sizeof(T);

    static constexpr uint32_t UB_AXES = (KEY / KEY_BASE) % KEY_BASE; // TilingKey倒数第二维为UB内轴个数
    static constexpr uint32_t MODE = (KEY / 1000) % KEY_BASE;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    TBuf<TPosition::VECCALC> inQueue_;
    TBuf<TPosition::VECCALC> outQueue_;
    TBuf<TPosition::VECCALC> idxQueue_;

    TPipe* pipe_ = nullptr;
    int64_t blockIdx_;
    uint32_t inTileSize_{0};
    uint32_t outTileSize_{0};
    uint32_t additionOffset_{0};
    const PadACTilingData* tilingData_ = nullptr;
    bool has1DPadding{true};
    bool needFlip{true};
    bool lastOut_{true};
    uint32_t padHLength_{1};
    uint32_t padWOutLength_{0};
    uint32_t padWInLength_{0};
    uint32_t padStride_[MAX_H_DIMS] = {0, 0, 0};
    uint16_t leftNum{0};
    uint16_t leftCrossNum{0};
    uint16_t middleNum{0};
    uint16_t rightCrossNum{0};
    uint16_t rightNum{0};
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t inCopyLen_[PAD_MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;
    constexpr static int32_t MAX_INDEX_NUM = 40;
    DataCopyExtParams copyOutParams;
    DataCopyExtParams copyOutParamsU;
    DataCopyExtParams copyOutParamsD;
    using RT = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    static constexpr uint32_t VL_ELEMS_R = VL_SIZE / sizeof(RT);
    static constexpr uint32_t VL_ELEMS_C = VL_SIZE / sizeof(CastType);

public:
    __aicore__ inline KernelPadMirrWithNormalWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();
        additionOffset_ = tilingData_->additionTileSize / sizeof(RT);
        inTileSize_ = tilingData_->outTileSize;
        outTileSize_ = tilingData_->outTileSize;
        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM * inTileSize_);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM * outTileSize_ * CONST2);
        pipe_->InitBuffer(idxQueue_, tilingData_->additionTileSize);

        const int8_t dimNum = tilingData_->dimNum;
        const int8_t ubAxis = tilingData_->ubAxis;
        padWOutLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->outShape[dimNum - 1]), BLK_ELEMS);
        padWInLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->inShape[dimNum - 1]), BLK_ELEMS);

        for (int8_t i = dimNum - 1; i > ubAxis; i--) {
            inCopyLen_[i] = tilingData_->inShape[i];
            if constexpr (UB_AXES > CONST2) {
                // UB内超过两个才需赋值下标1之后的参数
                if (i != dimNum - 1) {
                    padStride_[i - ubAxis] = padHLength_ * padWOutLength_;
                    padHLength_ = padHLength_ * tilingData_->outShape[i];
                }
            }
        }
        if (tilingData_->outShape[dimNum - 1] == tilingData_->inShape[dimNum - 1]) {
            has1DPadding = false;
        }
        padStride_[0] = padHLength_ * padWOutLength_;
        padHLength_ = padHLength_ * tilingData_->ubFactor;
    }

    __aicore__ inline void Process()
    {
        uint8_t ubAxis = tilingData_->ubAxis;
        uint64_t ubFactor = tilingData_->ubFactor;
        uint64_t ubPerCount = tilingData_->ubPerCount;
        uint64_t ubTotalCount = tilingData_->ubTotalCount;

        uint32_t startIdx = blockIdx_ * ubPerCount;
        if (startIdx >= ubTotalCount) {
            return;
        }
        uint32_t endIdx = (blockIdx_ + 1L) * ubPerCount;
        endIdx = (endIdx < ubTotalCount ? endIdx : ubTotalCount);

        PadMirrNormalParam padParam = {.padH = padHLength_,
                                       .padWI = padWInLength_,
                                       .padWO = padWOutLength_,
                                       .padLeft = static_cast<uint32_t>(tilingData_->leftPad[tilingData_->dimNum - 1]),
                                       .padRight = static_cast<uint32_t>(tilingData_->leftPad[tilingData_->dimNum - 1] +
                                                                         tilingData_->inShape[tilingData_->dimNum - 1]),
                                       .padStride = {padStride_[0], padStride_[1], padStride_[2]}};
        LocalTensor<RT> idxDst = idxQueue_.Get<RT>();
        if (has1DPadding) {
            CalPadPara(idxDst, padParam);
        }
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            lastOut_ = (idx == endIdx - 1);
            needFlip = true;
            uint32_t curIdx = idx;
            for (int32_t i = ubAxis; i >= 0; i--) {
                uint64_t factor = tilingData_->inShape[i];
                if (i == ubAxis) {
                    factor = CeilDiv(tilingData_->inShape[i], ubFactor);
                }
                inIndex_[i] = (i == ubAxis ? curIdx % factor * ubFactor : curIdx % factor);
                outIndex_[i] = inIndex_[i] + tilingData_->leftPad[i];
                curIdx = curIdx / factor;
            }
            inCopyLen_[ubAxis] = inIndex_[ubAxis] + ubFactor < tilingData_->inShape[ubAxis] ?
                                     ubFactor :
                                     tilingData_->inShape[ubAxis] - inIndex_[ubAxis];
            if (inCopyLen_[ubAxis] == 1) {
                needFlip = false;
            }
            ProcessOneStep(idxDst, idx - startIdx, padParam);
        }
    }

private:
    __aicore__ inline void ProcessOneStep(const LocalTensor<RT>& idxDst, int32_t idx, PadMirrNormalParam& padParam)
    {
        // Copy IN
        LocalTensor<T> input = inQueue_.Get<T>();
        LocalTensor<T> output = outQueue_.Get<T>();
        LocalTensor<T> srcLocal = input[(idx & 1) * inTileSize_ / sizeof(T)];
        LocalTensor<T> dstLocal = output[(idx & 1) * outTileSize_ * CONST2 / sizeof(T)];
        LocalTensor<T> dstFLocal = output[(idx & 1) * outTileSize_ * CONST2 / sizeof(T) + outTileSize_ / sizeof(T)];
        const int8_t dimNum = tilingData_->dimNum;
        CopyIn(srcLocal, padParam, idx);
        ProcessPad(srcLocal, dstLocal, dstFLocal, idxDst, padParam, idx);
        CopyOut(dstLocal, dstFLocal, padParam, idx);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadMirrNormalParam& padParam, int32_t idx)
    {
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum; i++) {
            inAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = inCopyLen_[dimNum - CONST2];
        copyInParams.blockLen = tilingData_->inShape[dimNum - 1] * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        if constexpr (UB_AXES == CONST3) {
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = inCopyLen_[dimNum - CONST3];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum - CONST3] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->inShape[dimNum - CONST2] * padParam.padWI * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src, input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else if constexpr (UB_AXES == CONST4) {
            LoopModeParams loopParams;
            loopParams.loop1Size = inCopyLen_[dimNum - CONST3];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum - CONST3] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->inShape[dimNum - CONST2] * padParam.padWI * sizeof(T);
            loopParams.loop2Size = inCopyLen_[dimNum - CONST4];
            loopParams.loop2SrcStride = tilingData_->inStride[dimNum - CONST4] * sizeof(T);
            loopParams.loop2DstStride = tilingData_->inShape[dimNum - CONST3] * tilingData_->inShape[dimNum - CONST2] *
                                        padParam.padWI * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src, input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else {
            DataCopyPad(src, input_[inAddr], copyInParams, padParams);
        }
    }

    __aicore__ inline void ProcessPad(const LocalTensor<T>& src, const LocalTensor<T>& dst, const LocalTensor<T>& dstF,
                                      const LocalTensor<RT>& idxDst, PadMirrNormalParam& padParam, int32_t idx)
    {
        SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }
        }
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        uint64_t outAddr = 0;
        for (uint32_t i = ubAxis + 1; i < dimNum - 1; i++) {
            outAddr += tilingData_->leftPad[i] * padParam.padStride[i - ubAxis];
        }
        if (has1DPadding) {
            GatherProcess(dst[outAddr], src, idxDst, padParam);
        } else {
            MoveProcess(dst[outAddr], src, padParam);
        }
        if constexpr (UB_AXES > CONST2) {
            outAddr -= tilingData_->leftPad[dimNum - CONST2] * padParam.padStride[dimNum - CONST2 - ubAxis];
            Pad2DProcess(dst[outAddr], padParam);
        }
        if constexpr (UB_AXES > CONST3) {
            outAddr -= tilingData_->leftPad[dimNum - CONST3] * padParam.padStride[dimNum - CONST3 - ubAxis];
            Pad3DProcess(dst[outAddr], padParam);
        }
        if (needFlip) {
            FlipProcess(dstF, dst, padParam);
        }
    }

    __aicore__ inline void CalPadPara(const LocalTensor<RT>& dst, PadMirrNormalParam& padParam)
    {
        leftNum = padParam.padLeft / VL_ELEMS_R;
        leftCrossNum = padParam.padLeft % VL_ELEMS_R == 0 ? 0 : 1;
        rightCrossNum = (padParam.padRight % VL_ELEMS_R > 0) ? 1 : 0;
        middleNum = padParam.padRight / VL_ELEMS_R - padParam.padLeft / VL_ELEMS_R > leftCrossNum ?
                        padParam.padRight / VL_ELEMS_R - padParam.padLeft / VL_ELEMS_R - leftCrossNum :
                        0;
        rightNum = padParam.padWO / VL_ELEMS_R - padParam.padRight / VL_ELEMS_R - rightCrossNum +
                   (padParam.padWO % VL_ELEMS_R == 0 ? 0 : 1);
        GenIndex(dst, padParam);
    }

    __aicore__ inline void GenIndex(const LocalTensor<RT>& dst, PadMirrNormalParam& padParam)
    {
        auto dstAddr = reinterpret_cast<__local_mem__ RT*>(dst.GetPhyAddr());
        const int8_t dimNum = tilingData_->dimNum;
        const uint16_t hasRight = rightNum > 0 ? 1 : 0;
        const uint16_t noLastRight = rightNum > 1 ? rightNum - 1 : 0;
        const uint16_t modeOffset = MODE <= 1 ? 0 : 1;
        const int32_t firstIndexLeft = padParam.padLeft - VL_ELEMS_R - modeOffset;
        const int32_t firstIndexMiddle = 0 - padParam.padLeft % VL_ELEMS_R;
        const int32_t firstIndexRight = rightCrossNum == 0 ?
                                            padParam.padRight - padParam.padLeft - 2 + modeOffset :
                                            padParam.padRight - padParam.padLeft + padParam.padRight % VL_ELEMS_R - 2 -
                                                VL_ELEMS_R + modeOffset;
        const uint32_t leftMaskLen = padParam.padLeft % VL_ELEMS_R;
        const uint32_t rightMaskLen = padParam.padRight % VL_ELEMS_R;
        const uint32_t endMaskLen = (tilingData_->outShape[dimNum - 1] % VL_ELEMS_R > 0 && rightNum <= 1) ?
                                        tilingData_->outShape[dimNum - 1] % VL_ELEMS_R :
                                        VL_ELEMS_R;
        const uint32_t endRMaskLen = (rightNum == 0) ? endMaskLen : VL_ELEMS_R;
        const uint32_t middleOffset = padParam.padRight / VL_ELEMS_R - padParam.padLeft / VL_ELEMS_R < leftCrossNum ?
                                          1 :
                                          0;
        const uint32_t endLMaskLen = (middleOffset > 0) ? endRMaskLen : VL_ELEMS_R;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> leftIdx;
            AscendC::Reg::RegTensor<RT> middleIdx;
            AscendC::Reg::RegTensor<RT> rightIdx;
            AscendC::Reg::RegTensor<RT> tempIdx;
            AscendC::Reg::MaskReg maskReg;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<RT, AscendC::Reg::MaskPattern::ALL>();
            uint32_t idOffset = 0;
            uint32_t leftLen = leftMaskLen;
            uint32_t rightLen = rightMaskLen;
            uint32_t midLen = (1 - middleOffset) * VL_ELEMS_R;
            uint32_t endLen = endMaskLen;
            uint32_t endRLen = endRMaskLen;
            uint32_t endLLen = endLMaskLen;
            RT addsScale = VL_ELEMS_R;
            Reg::Arange<RT, AscendC::Reg::IndexOrder::DECREASE_ORDER>(leftIdx, firstIndexLeft + 1);
            Reg::Arange<RT, AscendC::Reg::IndexOrder::DECREASE_ORDER>(tempIdx, firstIndexLeft + 1);
            Reg::Arange(middleIdx, firstIndexMiddle);
            Reg::Arange<RT, AscendC::Reg::IndexOrder::DECREASE_ORDER>(rightIdx, firstIndexRight + 1);

            for (uint16_t i = 0; i < leftNum; i++) {
                AscendC::Reg::DataCopy(dstAddr + idOffset * VL_ELEMS_R, tempIdx, maskAll);
                Adds(tempIdx, tempIdx, -1 * addsScale, maskAll);
                idOffset += 1;
            }
            for (uint16_t i = 0; i < leftCrossNum; i++) {
                maskReg = AscendC::Reg::UpdateMask<RT>(leftLen);
                AscendC::Reg::MaskNot(maskReg, maskReg, maskAll);
                Copy(tempIdx, middleIdx, maskReg);
                maskReg = AscendC::Reg::UpdateMask<RT>(endLLen);
                AscendC::Reg::DataCopy(dstAddr + idOffset * VL_ELEMS_R, tempIdx, maskReg);
                AscendC::Reg::Adds(middleIdx, middleIdx, addsScale, maskAll);
                idOffset += (1 - middleOffset);
            }
            maskReg = AscendC::Reg::UpdateMask<RT>(midLen);
            // leftcross和rightcross在同一block时，不copy mid
            Copy(tempIdx, middleIdx, maskReg);
            for (uint16_t i = 0; i < middleNum; i++) {
                AscendC::Reg::DataCopy(dstAddr + idOffset * VL_ELEMS_R, tempIdx, maskAll);
                AscendC::Reg::Adds(tempIdx, tempIdx, addsScale, maskAll);
                idOffset += 1;
            }
            for (uint16_t i = 0; i < rightCrossNum; i++) {
                maskReg = AscendC::Reg::UpdateMask<RT>(rightLen);
                AscendC::Reg::MaskNot(maskReg, maskReg, maskAll);
                Copy(tempIdx, rightIdx, maskReg);
                maskReg = AscendC::Reg::UpdateMask<RT>(endRLen);
                AscendC::Reg::DataCopy(dstAddr + idOffset * VL_ELEMS_R, tempIdx, maskReg);
                idOffset += 1;
            }
            for (uint16_t i = 0; i < noLastRight; i++) {
                AscendC::Reg::Adds(rightIdx, rightIdx, -1 * addsScale, maskAll);
                AscendC::Reg::DataCopy(dstAddr + idOffset * VL_ELEMS_R, rightIdx, maskAll);
                idOffset += 1;
            }
            for (uint16_t i = 0; i < hasRight; i++) {
                maskReg = AscendC::Reg::UpdateMask<RT>(endLen);
                AscendC::Reg::Adds(rightIdx, rightIdx, -1 * addsScale, maskReg);
                AscendC::Reg::DataCopy(dstAddr + idOffset * VL_ELEMS_R, rightIdx, maskReg);
            }
        }
    }

    __aicore__ inline void GatherProcess(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                         const LocalTensor<RT>& idx, PadMirrNormalParam& padParam)
    {
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        auto srcAddr = reinterpret_cast<__local_mem__ T*>(src.GetPhyAddr());
        auto idxAddr = reinterpret_cast<__local_mem__ RT*>(idx.GetPhyAddr());
        const int8_t dimNum = tilingData_->dimNum;
        const uint16_t padVLNum = tilingData_->outShape[dimNum - 1] / VL_ELEMS_C;
        const uint16_t endMaskLen = tilingData_->outShape[dimNum - 1] % VL_ELEMS_C;
        const uint16_t padBLNum = endMaskLen > 0 ? 1 : 0;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimNum - CONST4];
        const uint16_t dimCNum = (UB_AXES < CONST3) ? 1 : inCopyLen_[dimNum - CONST3];
        const uint16_t dimHNum = inCopyLen_[dimNum - CONST2];
        const uint32_t padWI = padParam.padWI;
        const uint32_t padWO = padParam.padWO;
        const uint32_t padHW = tilingData_->outShape[dimNum - CONST2] * padWO;
        const uint32_t padCHW = (UB_AXES > CONST3) ? tilingData_->outShape[dimNum - CONST3] * padHW : 0;

        __VEC_SCOPE__
        {
            Reg::MaskReg endMask;
            Reg::MaskReg maskAll;
            uint32_t endLen = endMaskLen * (VL_ELEMS / VL_ELEMS_C);
            endMask = AscendC::Reg::UpdateMask<T>(endLen);
            maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

            if constexpr (UB_AXES == CONST2) {
                for (uint16_t i = 0; i < dimHNum; i++) {
                    GatherProcessLine(dstAddr + i * padWO, srcAddr + i * padWI, idxAddr, padVLNum, padBLNum, endMask,
                                      maskAll);
                }
            } else if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    for (uint16_t i = 0; i < dimHNum; i++) {
                        GatherProcessLine(dstAddr + c * padHW + i * padWO, srcAddr + (c * dimHNum + i) * padWI, idxAddr,
                                          padVLNum, padBLNum, endMask, maskAll);
                    }
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        for (uint16_t i = 0; i < dimHNum; i++) {
                            GatherProcessLine(dstAddr + n * padCHW + c * padHW + i * padWO,
                                              srcAddr + (n * dimCNum * dimHNum + c * dimHNum + i) * padWI, idxAddr,
                                              padVLNum, padBLNum, endMask, maskAll);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void GatherProcessLine(__local_mem__ T* dstAddr, __local_mem__ T* srcAddr,
                                             __local_mem__ RT* idxAddr, uint16_t padVLNum, uint16_t padBLNum,
                                             Reg::MaskReg endMask, Reg::MaskReg maskAll)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<RT> idxTmp;
            AscendC::Reg::RegTensor<T> dataTmp;
            AscendC::Reg::RegTensor<T> dataT;

            for (uint16_t i = 0; i < padVLNum; i++) {
                AscendC::Reg::DataCopy(idxTmp, idxAddr + i * VL_ELEMS_C);
                AscendC::Reg::DataCopyGather((Reg::RegTensor<CastType>&)dataTmp, srcAddr,
                                             (Reg::RegTensor<IdxType>&)idxTmp, maskAll);
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopy(dstAddr + i * VL_ELEMS_C, dataTmp, maskAll);
                } else {
                    Reg::Pack(dataT, (Reg::RegTensor<CastType>&)dataTmp);
                    Reg::DataCopy(dstAddr + i * VL_ELEMS_C, dataT, maskAll);
                }
            }
            for (uint16_t i = 0; i < padBLNum; i++) {
                AscendC::Reg::DataCopy(idxTmp, idxAddr + padVLNum * VL_ELEMS_C);
                AscendC::Reg::DataCopyGather((Reg::RegTensor<CastType>&)dataTmp, srcAddr,
                                             (Reg::RegTensor<IdxType>&)idxTmp, endMask);
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopy(dstAddr + padVLNum * VL_ELEMS_C, dataTmp, endMask);
                } else {
                    Reg::Pack(dataT, (Reg::RegTensor<CastType>&)dataTmp);
                    Reg::DataCopy(dstAddr + padVLNum * VL_ELEMS_C, dataT, endMask);
                }
            }
        }
    }

    __aicore__ inline void MoveProcess(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                       PadMirrNormalParam& padParam)
    {
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        auto srcAddr = reinterpret_cast<__local_mem__ T*>(src.GetPhyAddr());
        const int8_t dimNum = tilingData_->dimNum;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimNum - CONST4];
        const uint16_t dimCNum = (UB_AXES < CONST3) ? 1 : inCopyLen_[dimNum - CONST3];
        const uint16_t dimHNum = inCopyLen_[dimNum - CONST2];
        const uint32_t padW = padParam.padWI;
        const uint32_t padHW = tilingData_->outShape[dimNum - CONST2] * padW;
        const uint32_t padCHW = (UB_AXES > CONST3) ? tilingData_->outShape[dimNum - CONST3] * padHW : 0;
        // 因为尾轴无pad，可一次move一面
        const uint32_t moveLen = dimHNum * padW;
        const uint16_t padVLNum = moveLen / VL_ELEMS;
        const uint16_t padBLNum = moveLen % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            Reg::MaskReg endMask;
            uint32_t endLen = padBLNum;
            endMask = AscendC::Reg::UpdateMask<T>(endLen);

            if constexpr (UB_AXES == CONST2) {
                MoveLine(dstAddr, srcAddr, padVLNum, BLNum, endMask);
            } else if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    MoveLine(dstAddr + c * padHW, srcAddr + c * moveLen, padVLNum, BLNum, endMask);
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        MoveLine(dstAddr + c * padHW + n * padCHW, srcAddr + c * moveLen + n * dimCNum * moveLen,
                                 padVLNum, BLNum, endMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void MoveLine(__local_mem__ T* dstAddr, __local_mem__ T* srcAddr, uint16_t padVLNum,
                                    uint16_t padBLNum, Reg::MaskReg endMask)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> vRegTmp;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
            for (uint16_t i = 0; i < padVLNum; i++) {
                AscendC::Reg::DataCopy(vRegTmp, srcAddr + i * VL_ELEMS);
                AscendC::Reg::DataCopy(dstAddr + i * VL_ELEMS, vRegTmp, maskAll);
            }
            for (uint16_t i = 0; i < padBLNum; i++) {
                AscendC::Reg::DataCopy(vRegTmp, srcAddr + padVLNum * VL_ELEMS);
                AscendC::Reg::DataCopy(dstAddr + padVLNum * VL_ELEMS, vRegTmp, endMask);
            }
        }
    }

    __aicore__ inline void Pad2DProcess(const LocalTensor<T>& dst, PadMirrNormalParam& padParam)
    {
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        const int8_t dimNum = tilingData_->dimNum;
        const uint16_t modeOffset = MODE <= 1 ? 0 : 1;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimNum - CONST4];
        const uint16_t dimCNum = inCopyLen_[dimNum - CONST3];
        const uint32_t padWO = padParam.padWO;
        const uint32_t padHW = tilingData_->outShape[dimNum - CONST2] * padWO;
        const uint32_t padCHW = (UB_AXES > CONST3) ? tilingData_->outShape[dimNum - CONST3] * padHW : 0;
        const uint16_t upNum = tilingData_->leftPad[dimNum - CONST2];
        const uint16_t upOffset = upNum >= modeOffset ? upNum * 2 - modeOffset : 0;
        const uint16_t downNum = tilingData_->outShape[dimNum - CONST2] - tilingData_->inShape[dimNum - CONST2] -
                                 tilingData_->leftPad[dimNum - CONST2];
        const uint16_t downOffset = upNum + tilingData_->inShape[dimNum - CONST2] + modeOffset > 2 ?
                                        upNum + tilingData_->inShape[dimNum - CONST2] + modeOffset - 2 :
                                        0;
        const uint16_t downInOffset = upNum + tilingData_->inShape[dimNum - CONST2];
        const uint16_t padVLNum = padWO / VL_ELEMS;
        const uint16_t padBLNum = padWO % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            Reg::MaskReg endMask;
            uint32_t endLen = padBLNum;
            endMask = AscendC::Reg::UpdateMask<T>(endLen);

            if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    for (uint16_t i = 0; i < upNum; i++) {
                        MoveLine(dstAddr + c * padHW + i * padWO, dstAddr + c * padHW + (upOffset - i) * padWO,
                                 padVLNum, BLNum, endMask);
                    }
                    for (uint16_t i = 0; i < downNum; i++) {
                        MoveLine(dstAddr + c * padHW + (i + downInOffset) * padWO,
                                 dstAddr + c * padHW + (downOffset - i) * padWO, padVLNum, BLNum, endMask);
                    }
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        uint32_t tempOffset = c * padHW + n * padCHW;
                        for (uint16_t i = 0; i < upNum; i++) {
                            MoveLine(dstAddr + tempOffset + i * padWO, dstAddr + tempOffset + (upOffset - i) * padWO,
                                     padVLNum, BLNum, endMask);
                        }
                        for (uint16_t i = 0; i < downNum; i++) {
                            MoveLine(dstAddr + tempOffset + (i + downInOffset) * padWO,
                                     dstAddr + tempOffset + (downOffset - i) * padWO, padVLNum, BLNum, endMask);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void Pad3DProcess(const LocalTensor<T>& dst, PadMirrNormalParam& padParam)
    {
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        const int8_t dimNum = tilingData_->dimNum;
        const uint16_t modeOffset = MODE <= 1 ? 0 : 1;
        const uint16_t dimNNum = inCopyLen_[dimNum - CONST4];
        const uint32_t padHW = tilingData_->outShape[dimNum - CONST2] * padParam.padWO;
        const uint32_t padCHW = tilingData_->outShape[dimNum - CONST3] * padHW;
        const uint16_t upNum = tilingData_->leftPad[dimNum - CONST3];
        const uint16_t upOffset = upNum >= modeOffset ? upNum * 2 - modeOffset : 0;
        const uint16_t downNum = tilingData_->outShape[dimNum - CONST3] - tilingData_->inShape[dimNum - CONST3] -
                                 tilingData_->leftPad[dimNum - CONST3];
        const uint16_t downOffset = upNum + tilingData_->inShape[dimNum - CONST3] + modeOffset > 2 ?
                                        upNum + tilingData_->inShape[dimNum - CONST3] + modeOffset - 2 :
                                        0;
        const uint16_t downInOffset = upNum + tilingData_->inShape[dimNum - CONST3];
        const uint16_t padVLNum = padHW / VL_ELEMS;
        const uint16_t padBLNum = padHW % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            Reg::MaskReg endMask;
            uint32_t endLen = padBLNum;
            endMask = AscendC::Reg::UpdateMask<T>(endLen);

            for (uint16_t n = 0; n < dimNNum; n++) {
                for (uint16_t i = 0; i < upNum; i++) {
                    MoveLine(dstAddr + n * padCHW + i * padHW, dstAddr + n * padCHW + (upOffset - i) * padHW, padVLNum,
                             BLNum, endMask);
                }
                for (uint16_t i = 0; i < downNum; i++) {
                    MoveLine(dstAddr + n * padCHW + (i + downInOffset) * padHW,
                             dstAddr + n * padCHW + (downOffset - i) * padHW, padVLNum, BLNum, endMask);
                }
            }
        }
    }

    __aicore__ inline void FlipProcess(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                       PadMirrNormalParam& padParam)
    {
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        auto srcAddr = reinterpret_cast<__local_mem__ T*>(src.GetPhyAddr());
        const int8_t dimNum = tilingData_->dimNum;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimNum - CONST4];
        const uint16_t dimCNum = (UB_AXES < CONST3) ? 1 : inCopyLen_[dimNum - CONST3];
        const uint16_t dimHNum = inCopyLen_[dimNum - CONST2];
        const uint32_t padHW = tilingData_->outShape[dimNum - CONST2] * padParam.padWO;
        const uint32_t padCHW = (UB_AXES < CONST4) ? padHW : tilingData_->outShape[dimNum - CONST3] * padHW;
        const uint32_t flipLen = (UB_AXES < CONST3) ? padParam.padWO : (UB_AXES < CONST4) ? padHW : padCHW;
        const uint16_t totalNum = (UB_AXES < CONST3) ? dimHNum : (UB_AXES < CONST4) ? dimCNum : dimNNum;
        const uint16_t padVLNum = flipLen / VL_ELEMS;
        const uint16_t padBLNum = flipLen % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> vRegTmp;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::MaskReg outMask;
            uint32_t outLen = padBLNum;
            outMask = AscendC::Reg::UpdateMask<T>(outLen);

            for (uint16_t n = 0; n < totalNum; n++) {
                MoveLine(dstAddr + n * flipLen, srcAddr + (totalNum - 1 - n) * flipLen, padVLNum, BLNum, outMask);
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, const LocalTensor<T>& srcT, PadMirrNormalParam& padParam,
                                   int32_t idx)
    {
        SetEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            }
        }
        if (idx >= 1) {
            int32_t nextIdx = idx + 1;
            if ((nextIdx & 1) == 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
        const int8_t ubAxis = tilingData_->ubAxis;
        int64_t Indexes[MAX_INDEX_NUM] = {0};
        int32_t curFromIndex = 0;
        int32_t curToIndex = 1;
        for (auto i = ubAxis - 1; i >= 0; i--) {
            int32_t curOffset = 0;
            int64_t upIndex = (tilingData_->leftPad[i] > 0) ? calOutIndex(i, true) : -1;
            int64_t downIndex = (tilingData_->leftPad[i] + tilingData_->inShape[i] < tilingData_->outShape[i]) ?
                                    calOutIndex(i, false) :
                                    -1;
            for (auto j = curFromIndex; j < curToIndex; j++) {
                if (upIndex >= 0) {
                    Indexes[curToIndex + curOffset] = upIndex * tilingData_->outStride[i] + Indexes[j];
                    curOffset += 1;
                }
                Indexes[curToIndex + curOffset] = outIndex_[i] * tilingData_->outStride[i] + Indexes[j];
                curOffset += 1;
                if (downIndex > 0) {
                    Indexes[curToIndex + curOffset] = downIndex * tilingData_->outStride[i] + Indexes[j];
                    curOffset += 1;
                }
            }
            curFromIndex = curToIndex;
            curToIndex = curToIndex + curOffset;
        }
        CopyOutSelf(src, Indexes, padParam, curFromIndex, curToIndex);
        if (needFlip) {
            CopyOutPad(srcT, Indexes, padParam, curFromIndex, curToIndex);
        } else {
            CopyOutPad(src, Indexes, padParam, curFromIndex, curToIndex);
        }
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void CopyOutSelf(const LocalTensor<T>& dst, int64_t (&Indexes)[MAX_INDEX_NUM],
                                       PadMirrNormalParam& padParam, int32_t curFromIndex, int32_t curToIndex)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        uint64_t outAddr = 0;
        uint32_t blockCount = inCopyLen_[ubAxis];
        for (auto i = ubAxis + 1; i < dimNum - 1; i++) {
            // blockCount 计算到倒数第二维
            blockCount = blockCount * tilingData_->outShape[i];
        }
        copyOutParams.blockCount = blockCount;
        copyOutParams.blockLen = tilingData_->outShape[dimNum - 1] * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        int16_t max_index = 0;
        if (ubAxis == 0) {
            for (uint32_t i = 0; i <= ubAxis; i++) {
                outAddr += outIndex_[i] * tilingData_->outStride[i];
            }
            DataCopyPad(output_[outAddr], dst, copyOutParams);
        } else {
            outAddr = outIndex_[ubAxis] * tilingData_->outStride[ubAxis];
            for (auto i = curFromIndex; i < curToIndex; i++) {
                DataCopyPad(output_[Indexes[i] + outAddr], dst, copyOutParams);
            }
        }
    }

    __aicore__ inline int64_t calOutIndex(int8_t curAxis, bool isUp)
    {
        int64_t res = -1;
        uint16_t modeOffset = MODE <= 1 ? 0 : 1;
        if (isUp) {
            uint64_t first = tilingData_->leftPad[curAxis] - modeOffset;
            uint64_t last = 1 - modeOffset;
            if (inIndex_[curAxis] <= first && inIndex_[curAxis] >= last) {
                res = first - inIndex_[curAxis];
            }
        } else {
            uint64_t first = tilingData_->inShape[curAxis] + modeOffset - 2;
            uint64_t last = tilingData_->leftPad[curAxis] + tilingData_->inShape[curAxis] * 2 + modeOffset -
                            tilingData_->outShape[curAxis] - 1;
            if (inIndex_[curAxis] <= first && inIndex_[curAxis] >= last) {
                res = tilingData_->leftPad[curAxis] + tilingData_->inShape[curAxis] + first - inIndex_[curAxis];
            }
        }
        return res;
    }

    __aicore__ inline void CopyOutPad(const LocalTensor<T>& dstT, int64_t (&Indexes)[MAX_INDEX_NUM],
                                      PadMirrNormalParam& padParam, int32_t curFromIndex, int32_t curToIndex)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        uint16_t modeOffset = MODE <= 1 ? 0 : 1;
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < ubAxis; i++) {
            outAddr += outIndex_[i] * tilingData_->outStride[i];
        }
        uint32_t first = inIndex_[ubAxis];
        uint32_t last = inIndex_[ubAxis] + inCopyLen_[ubAxis];
        uint32_t blockCountU = tilingData_->leftPad[ubAxis] - modeOffset > last - 1 ?
                                   inCopyLen_[ubAxis] :
                                   (tilingData_->leftPad[ubAxis] - modeOffset - first + 1 > 0 ?
                                        tilingData_->leftPad[ubAxis] - modeOffset - first + 1 :
                                        0);
        blockCountU = (first < 1 - modeOffset && blockCountU > 0) ? blockCountU - 1 : blockCountU;
        for (auto i = ubAxis + 1; i < dimNum - 1; i++) {
            // blockCount 计算到倒数第二维
            blockCountU = blockCountU * tilingData_->outShape[i];
        }
        copyOutParamsU.blockCount = blockCountU;
        copyOutParamsU.blockLen = tilingData_->outShape[dimNum - 1] * sizeof(T);
        copyOutParamsU.srcStride = 0;
        copyOutParamsU.dstStride = 0;
        uint32_t lastIndex = tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis] * 2 + modeOffset -
                             tilingData_->outShape[ubAxis] - 1;
        uint32_t blockCountD = Std::min(last, static_cast<uint32_t>(tilingData_->inShape[ubAxis] + modeOffset - 1)) >
                                       Std::max(lastIndex, first) ?
                                   (Std::min(last,
                                             static_cast<uint32_t>(tilingData_->inShape[ubAxis] + modeOffset - 1)) -
                                    Std::max(lastIndex, first)) :
                                   0;
        for (auto i = ubAxis + 1; i < dimNum - 1; i++) {
            // blockCount 计算到倒数第二维
            blockCountD = blockCountD * tilingData_->outShape[i];
        }
        copyOutParamsD.blockCount = blockCountD;
        copyOutParamsD.blockLen = tilingData_->outShape[dimNum - 1] * sizeof(T);
        copyOutParamsD.srcStride = 0;
        copyOutParamsD.dstStride = 0;

        if (copyOutParamsU.blockCount > 0) {
            uint64_t upInOffset = tilingData_->leftPad[ubAxis] - modeOffset > last - 1 ?
                                      0 :
                                      (last - 1 - tilingData_->leftPad[ubAxis] + modeOffset);
            uint64_t upOutOffset = tilingData_->leftPad[ubAxis] - modeOffset > last - 1 ?
                                       (tilingData_->leftPad[ubAxis] - modeOffset - last + 1) :
                                       0;
            if (ubAxis == 0) {
                DataCopyPad(output_[outAddr + upOutOffset * tilingData_->outStride[ubAxis]],
                            dstT[upInOffset * padParam.padStride[0]], copyOutParamsU);
            } else {
                for (auto i = curFromIndex; i < curToIndex; i++) {
                    DataCopyPad(output_[Indexes[i] + upOutOffset * tilingData_->outStride[ubAxis]],
                                dstT[upInOffset * padParam.padStride[0]], copyOutParamsU);
                }
            }
        }
        if (copyOutParamsD.blockCount > 0) {
            uint64_t downInOffset = (last == tilingData_->inShape[ubAxis]) ? (1 - modeOffset) : 0;
            uint64_t downOutOffset = (tilingData_->leftPad[ubAxis] + tilingData_->inShape[ubAxis] * 2 - last - 1 +
                                      modeOffset + downInOffset);
            if (ubAxis == 0) {
                DataCopyPad(output_[outAddr + downOutOffset * tilingData_->outStride[ubAxis]],
                            dstT[downInOffset * padParam.padStride[0]], copyOutParamsD);
            } else {
                for (auto i = curFromIndex; i < curToIndex; i++) {
                    DataCopyPad(output_[Indexes[i] + downOutOffset * tilingData_->outStride[ubAxis]],
                                dstT[downInOffset * padParam.padStride[0]], copyOutParamsD);
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
