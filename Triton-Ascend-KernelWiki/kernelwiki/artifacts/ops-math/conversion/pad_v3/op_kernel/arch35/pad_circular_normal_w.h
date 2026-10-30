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
 * \file pad_circular_normal_w.h
 * \brief pad cut not last dim kernel in circular mode
 */

#ifndef PAD_CIRC_NORMAL_W_H_
#define PAD_CIRC_NORMAL_W_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;
struct PadCircNormalParam {
    uint32_t padH;
    uint32_t padWO;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t padStride[MAX_H_DIMS];
};
template <typename T, int32_t KEY>
class KernelPadCircularWithNormalWidth {
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
    uint32_t outTileOffset_{0};
    uint32_t additionOffset_{0};
    const PadACTilingData* tilingData_ = nullptr;
    bool has1DPadding{true};
    bool lastOut_{true};
    uint32_t padHLength_{1};
    uint32_t padWOutLength_{0};
    uint32_t padStride_[MAX_H_DIMS] = {0, 0, 0};
    uint64_t inIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t inCopyLen_[PAD_MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t outIndex_[PAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;
    constexpr static int32_t MAX_INDEX_NUM = 40;
    constexpr static uint32_t HALF_SIZE = 2;
    uint64_t upOffset_{0};
    uint32_t blockCount_{0};
    uint32_t blockCountU_{0};
    uint32_t blockCountD_{0};

public:
    __aicore__ inline KernelPadCircularWithNormalWidth(TPipe* pipe, const PadACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y)
    {
        blockIdx_ = GetBlockIdx();
        additionOffset_ = tilingData_->additionTileSize / sizeof(T);
        outTileOffset_ = tilingData_->outTileSize / sizeof(T);
        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM * (tilingData_->outTileSize + tilingData_->additionTileSize));

        const int8_t dimNum = tilingData_->dimNum;
        const int8_t ubAxis = tilingData_->ubAxis;
        padWOutLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->outShape[dimNum - 1]), BLK_ELEMS);

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

        PadCircNormalParam padParam = {
            .padH = padHLength_,
            .padWO = padWOutLength_,
            .padLeft = static_cast<uint32_t>(padWOutLength_ - tilingData_->outShape[tilingData_->dimNum - 1] +
                                             tilingData_->leftPad[tilingData_->dimNum - 1]),
            .padRight = static_cast<uint32_t>(tilingData_->outShape[tilingData_->dimNum - 1] -
                                              tilingData_->leftPad[tilingData_->dimNum - 1] -
                                              tilingData_->inShape[tilingData_->dimNum - 1]),
            .padStride = {padStride_[0], padStride_[1], padStride_[2]}};

        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            lastOut_ = (idx == endIdx - 1);
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
            ProcessOneStep(idx - startIdx, padParam);
        }
    }

    __aicore__ inline void ProcessOneStep(int32_t idx, PadCircNormalParam& padParam)
    {
        LocalTensor<T> input = inQueue_.Get<T>();
        LocalTensor<T>
            srcLocal = input[(idx & 1) * (tilingData_->outTileSize + tilingData_->additionTileSize) / sizeof(T)];
        const int8_t dimNum = tilingData_->dimNum;
        CopyIn(srcLocal, padParam, idx);
        ProcessPad(srcLocal, padParam, idx);
        CopyOut(srcLocal, padParam, idx);
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, PadCircNormalParam& padParam, int32_t idx)
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
        copyInParams.dstStride = (padParam.padWO - tilingData_->inShape[dimNum - 1]) / BLK_ELEMS;

        uint64_t padHW = tilingData_->outShape[dimNum - CONST2] * padParam.padWO;
        uint32_t firstOffset3D = tilingData_->leftPad[dimNum - CONST2] * padParam.padWO;
        uint32_t firstOffset4D = tilingData_->leftPad[dimNum - CONST2] * padParam.padWO +
                                 tilingData_->leftPad[dimNum - CONST3] * padHW;

        if constexpr (UB_AXES == CONST3) {
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = inCopyLen_[dimNum - CONST3];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum - CONST3] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->outShape[dimNum - CONST2] * padParam.padWO * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[additionOffset_ + firstOffset3D], input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else if constexpr (UB_AXES == CONST4) {
            LoopModeParams loopParams;
            loopParams.loop1Size = inCopyLen_[dimNum - CONST3];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum - CONST3] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->outShape[dimNum - CONST2] * padParam.padWO * sizeof(T);
            loopParams.loop2Size = inCopyLen_[dimNum - CONST4];
            loopParams.loop2SrcStride = tilingData_->inStride[dimNum - CONST4] * sizeof(T);
            loopParams.loop2DstStride = tilingData_->outShape[dimNum - CONST3] *
                                        tilingData_->outShape[dimNum - CONST2] * padParam.padWO * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[additionOffset_ + firstOffset4D], input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else {
            DataCopyPad(src[additionOffset_], input_[inAddr], copyInParams, padParams);
        }
    }

    __aicore__ inline void ProcessPad(const LocalTensor<T>& src, PadCircNormalParam& padParam, int32_t idx)
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
        const int8_t dimNum = tilingData_->dimNum;
        const int8_t ubAxis = tilingData_->ubAxis;
        uint64_t outAddr = 0;
        for (uint32_t i = ubAxis + 1; i < dimNum - 1; i++) {
            outAddr += tilingData_->leftPad[i] * padParam.padStride[i - ubAxis];
        }

        PadRightSide(src, padParam, outAddr);
        PadLeftSide(src, padParam, outAddr, false);
        if constexpr (UB_AXES >= CONST3) {
            outAddr -= tilingData_->leftPad[dimNum - CONST2] * padParam.padWO;
            PadCopy(src, padParam, outAddr, dimNum - CONST2, true);
            PadCopy(src, padParam, outAddr, dimNum - CONST2, false);
        }
        if constexpr (UB_AXES >= CONST4) {
            outAddr -= tilingData_->leftPad[dimNum - CONST3] * padParam.padWO * tilingData_->outShape[dimNum - CONST2];
            PadCopy(src, padParam, outAddr, dimNum - CONST3, true);
            PadCopy(src, padParam, outAddr, dimNum - CONST3, false);
        }
        PadLeftSideFirst(src, src[additionOffset_], padParam);
        if constexpr (UB_AXES >= CONST3) {
            PadLeftSide(src, padParam, 0, true);
            if (tilingData_->leftPad[dimNum - CONST2]) {
                PadLeftSide(src, padParam, tilingData_->leftPad[dimNum - CONST2] * padParam.padWO, true);
            }
            if (tilingData_->leftPad[dimNum - CONST2] + tilingData_->inShape[dimNum - CONST2] <
                tilingData_->outShape[dimNum - CONST2]) {
                PadLeftSide(
                    src, padParam,
                    (tilingData_->leftPad[dimNum - CONST2] + tilingData_->inShape[dimNum - CONST2]) * padParam.padWO,
                    true);
            }
        }
        CalOutParam(padParam);
        if (upOffset_ > 0) {
            PadLeftSideFirst(src[additionOffset_ / HALF_SIZE], src[additionOffset_ + upOffset_], padParam);
        }
    }

    __aicore__ inline void CalOutParam(PadCircNormalParam& padParam)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        uint32_t first = inIndex_[ubAxis];
        uint32_t last = inIndex_[ubAxis] + inCopyLen_[ubAxis];
        uint32_t upStart = tilingData_->inShape[ubAxis] - tilingData_->leftPad[ubAxis];
        uint32_t downEnd = tilingData_->outShape[ubAxis] - tilingData_->inShape[ubAxis] - tilingData_->leftPad[ubAxis];
        blockCount_ = inCopyLen_[ubAxis];
        blockCountU_ = last <= upStart ? 0 : (first < upStart ? last - upStart : inCopyLen_[ubAxis]);
        blockCountD_ = first >= downEnd ? 0 : (last > downEnd ? downEnd - first : inCopyLen_[ubAxis]);
        for (auto i = ubAxis + 1; i < dimNum - 1; i++) {
            // blockCount 计算到倒数第二维
            blockCount_ = blockCount_ * tilingData_->outShape[i];
            blockCountU_ = blockCountU_ * tilingData_->outShape[i];
            blockCountD_ = blockCountD_ * tilingData_->outShape[i];
        }

        upOffset_ = (blockCountU_ == 0 || first >= upStart) ? 0 : (upStart - first) * padParam.padStride[0];
    }

    __aicore__ inline void PadLeftSideFirst(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                            PadCircNormalParam& padParam)
    {
        const int8_t dimNum = tilingData_->dimNum;
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        auto srcAddr = reinterpret_cast<__local_mem__ T*>(src.GetPhyAddr());
        const uint32_t moveLen = tilingData_->leftPad[dimNum - 1];
        const uint32_t InOffset = tilingData_->inShape[dimNum - 1] - tilingData_->leftPad[dimNum - 1];
        const uint16_t padVLNum = moveLen / VL_ELEMS;
        const uint16_t padBLNum = moveLen % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;

        __VEC_SCOPE__
        {
            Reg::MaskReg endMask;
            uint32_t endLen = padBLNum;
            endMask = AscendC::Reg::UpdateMask<T>(endLen);
            AscendC::Reg::RegTensor<T> vRegTmp;
            AscendC::Reg::UnalignReg uReg;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

            for (uint16_t i = 0; i < padVLNum; i++) {
                __ubuf__ T* Addr6 = srcAddr + InOffset + i * VL_ELEMS;
                AscendC::Reg::DataCopyUnAlignPre(uReg, Addr6);
                AscendC::Reg::DataCopyUnAlign(vRegTmp, uReg, Addr6);
                AscendC::Reg::DataCopy(dstAddr + i * VL_ELEMS, vRegTmp, maskAll);
            }
            for (uint16_t i = 0; i < BLNum; i++) {
                __ubuf__ T* Addr7 = srcAddr + InOffset + padVLNum * VL_ELEMS;
                AscendC::Reg::DataCopyUnAlignPre(uReg, Addr7);
                AscendC::Reg::DataCopyUnAlign(vRegTmp, uReg, Addr7);
                AscendC::Reg::DataCopy(dstAddr + padVLNum * VL_ELEMS, vRegTmp, endMask);
            }
        }
    }

    __aicore__ inline void PadLeftSide(const LocalTensor<T>& dst, PadCircNormalParam& padParam, uint32_t ubOffset,
                                       bool isFirst)
    {
        if (padParam.padLeft == 0) {
            return;
        }
        const int8_t dimNum = tilingData_->dimNum;
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        const uint32_t InOffset = tilingData_->inShape[dimNum - 1] - tilingData_->leftPad[dimNum - 1];
        const uint16_t padVLNum = padParam.padLeft / VL_ELEMS;
        const uint16_t padBLNum = padParam.padLeft % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;
        const uint32_t padW = padParam.padStride[UB_AXES - CONST2];
        const uint32_t padHW = (UB_AXES >= CONST3) ? padParam.padStride[UB_AXES - CONST3] : 0;
        const uint32_t padCHW = (UB_AXES >= CONST4) ? padParam.padStride[UB_AXES - CONST4] : 0;
        const uint16_t tempCNum = (UB_AXES < CONST4 ? 1 : inCopyLen_[dimNum - CONST4]) *
                                  (UB_AXES < CONST3 ? 1 :
                                                      (UB_AXES > CONST3 ? tilingData_->outShape[dimNum - CONST3] :
                                                                          inCopyLen_[dimNum - CONST3]));
        const uint16_t dimNNum = (UB_AXES < CONST4 || isFirst) ? 1 : inCopyLen_[dimNum - CONST4];
        const uint16_t dimCNum = isFirst ? tempCNum - (ubOffset > 0 ? 0 : 1) :
                                           (UB_AXES < CONST3 ? 1 : inCopyLen_[dimNum - CONST3]);
        const uint16_t dimHNum = isFirst ? 1 : inCopyLen_[dimNum - CONST2] - 1;
        const uint32_t startOffset = (isFirst ? (ubOffset > 0 ? ubOffset : ubOffset + padHW) : ubOffset + padW) +
                                     additionOffset_;

        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg lMask;
            uint32_t nolPadLen = VL_ELEMS - padBLNum;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
            lMask = AscendC::Reg::UpdateMask<T>(nolPadLen);
            AscendC::Reg::MaskNot(lMask, lMask, maskAll);

            if constexpr (UB_AXES == CONST2) {
                for (uint16_t h = 0; h < dimHNum; h++) {
                    PadLeftSideOne(dstAddr, startOffset + h * padW, InOffset, padVLNum, padBLNum, BLNum, lMask);
                }
            } else if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    for (uint16_t h = 0; h < dimHNum; h++) {
                        PadLeftSideOne(dstAddr, startOffset + c * padHW + h * padW, InOffset, padVLNum, padBLNum, BLNum,
                                       lMask);
                    }
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        for (uint16_t h = 0; h < dimHNum; h++) {
                            PadLeftSideOne(dstAddr, startOffset + n * padCHW + c * padHW + h * padW, InOffset, padVLNum,
                                           padBLNum, BLNum, lMask);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void PadLeftSideOne(__local_mem__ T* srcAddr, uint32_t firstOffset, uint32_t InOffset,
                                          uint16_t padVLNum, uint16_t padBLNum, uint16_t BLNum, Reg::MaskReg endMask)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> vRegTmp;
            AscendC::Reg::UnalignReg uReg;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();

            for (uint16_t i = 0; i < padVLNum; i++) {
                __ubuf__ T* Addr0 = srcAddr + firstOffset + InOffset + padBLNum + i * VL_ELEMS;
                AscendC::Reg::DataCopyUnAlignPre(uReg, Addr0);
                AscendC::Reg::DataCopyUnAlign(vRegTmp, uReg, Addr0);
                AscendC::Reg::DataCopy(srcAddr + firstOffset - (padVLNum - i) * VL_ELEMS, vRegTmp, maskAll);
            }
            for (uint16_t i = 0; i < BLNum; i++) {
                __ubuf__ T* Addr2 = srcAddr + firstOffset + InOffset + padBLNum - VL_ELEMS;
                AscendC::Reg::DataCopyUnAlignPre(uReg, Addr2);
                AscendC::Reg::DataCopyUnAlign(vRegTmp, uReg, Addr2);
                AscendC::Reg::DataCopy(srcAddr + firstOffset - (padVLNum + 1) * VL_ELEMS, vRegTmp, endMask);
            }
        }
    }

    __aicore__ inline void PadRightSide(const LocalTensor<T>& dst, PadCircNormalParam& padParam, uint32_t ubOffset)
    {
        if (padParam.padRight == 0) {
            return;
        }
        const int8_t dimNum = tilingData_->dimNum;
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        const uint32_t OutOffset = tilingData_->inShape[dimNum - 1];
        const uint16_t padVLNum = padParam.padRight / VL_ELEMS;
        const uint16_t padBLNum = padParam.padRight % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;
        const uint32_t padW = padParam.padStride[UB_AXES - CONST2];
        const uint32_t padHW = (UB_AXES >= CONST3) ? padParam.padStride[UB_AXES - CONST3] : 0;
        const uint32_t padCHW = (UB_AXES >= CONST4) ? padParam.padStride[UB_AXES - CONST4] : 0;
        const uint16_t dimNNum = (UB_AXES < CONST4) ? 1 : inCopyLen_[dimNum - CONST4];
        const uint16_t dimCNum = UB_AXES < CONST3 ? 1 : inCopyLen_[dimNum - CONST3];
        const uint16_t dimHNum = inCopyLen_[dimNum - CONST2];
        const uint32_t startOffset = ubOffset + additionOffset_;

        __VEC_SCOPE__
        {
            if constexpr (UB_AXES == CONST2) {
                for (uint16_t h = 0; h < dimHNum; h++) {
                    PadRightSideOne(dstAddr, startOffset + h * padW, OutOffset, padVLNum, padBLNum, BLNum);
                }
            } else if constexpr (UB_AXES == CONST3) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    for (uint16_t h = 0; h < dimHNum; h++) {
                        PadRightSideOne(dstAddr, startOffset + c * padHW + h * padW, OutOffset, padVLNum, padBLNum,
                                        BLNum);
                    }
                }
            } else if constexpr (UB_AXES == CONST4) {
                for (uint16_t n = 0; n < dimNNum; n++) {
                    for (uint16_t c = 0; c < dimCNum; c++) {
                        for (uint16_t h = 0; h < dimHNum; h++) {
                            PadRightSideOne(dstAddr, startOffset + n * padCHW + c * padHW + h * padW, OutOffset,
                                            padVLNum, padBLNum, BLNum);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void PadRightSideOne(__local_mem__ T* srcAddr, uint32_t firstOffset, uint32_t OutOffset,
                                           uint16_t padVLNum, uint16_t padBLNum, uint16_t BLNum)
    {
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> vRegTmp;
            AscendC::Reg::UnalignReg uReg;
            uint32_t padLen = padBLNum;
            uint32_t allLen = VL_ELEMS;

            for (uint16_t i = 0; i < padVLNum; i++) {
                __ubuf__ T* Addr3 = srcAddr + firstOffset + OutOffset + i * VL_ELEMS;
                AscendC::Reg::DataCopy(vRegTmp, srcAddr + firstOffset + i * VL_ELEMS);
                AscendC::Reg::DataCopyUnAlign(Addr3, vRegTmp, uReg, allLen);
                AscendC::Reg::DataCopyUnAlignPost(Addr3, uReg, 0);
            }
            for (uint16_t i = 0; i < BLNum; i++) {
                __ubuf__ T* Addr4 = srcAddr + firstOffset + OutOffset + padVLNum * VL_ELEMS;
                AscendC::Reg::DataCopy(vRegTmp, srcAddr + firstOffset + padVLNum * VL_ELEMS);
                AscendC::Reg::DataCopyUnAlign(Addr4, vRegTmp, uReg, padLen);
                AscendC::Reg::DataCopyUnAlignPost(Addr4, uReg, 0);
            }
        }
    }

    __aicore__ inline void PadCopy(const LocalTensor<T>& dst, PadCircNormalParam& padParam, uint32_t ubOffset,
                                   int8_t curAxis, bool isUp)
    {
        const int8_t ubAxis = tilingData_->ubAxis;
        const int8_t dimNum = tilingData_->dimNum;
        auto dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        const uint32_t padHW = padParam.padStride[curAxis - ubAxis - 1];
        const uint32_t padCHW = (curAxis - ubAxis <= 1) ? 0 : padParam.padStride[curAxis - ubAxis - 2];
        const uint32_t padW = padParam.padStride[curAxis - ubAxis];
        const uint16_t dimNNum = (curAxis - ubAxis <= 1) ? 1 : inCopyLen_[curAxis - CONST2];
        const uint16_t dimCNum = inCopyLen_[curAxis - 1];
        const uint32_t ucopyLen = (isUp ? tilingData_->leftPad[curAxis] * padW :
                                          (tilingData_->outShape[curAxis] - tilingData_->leftPad[curAxis] -
                                           tilingData_->inShape[curAxis]) *
                                              padW);
        const uint32_t copyLen = ucopyLen > padParam.padLeft ? ucopyLen - padParam.padLeft : 0;
        if (copyLen <= 0) {
            return;
        }
        const uint32_t InOffset = isUp ? tilingData_->inShape[curAxis] * padW : tilingData_->leftPad[curAxis] * padW;
        const uint32_t OutOffset = isUp ? 0 : (tilingData_->inShape[curAxis] + tilingData_->leftPad[curAxis]) * padW;
        const uint16_t padVLNum = copyLen / VL_ELEMS;
        const uint16_t padBLNum = copyLen % VL_ELEMS;
        const uint16_t BLNum = padBLNum > 0 ? 1 : 0;
        const uint32_t startOffset = ubOffset + additionOffset_;

        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg endMask;
            AscendC::Reg::RegTensor<T> vRegTmp;
            uint32_t PadLen = padBLNum;
            AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
            endMask = AscendC::Reg::UpdateMask<T>(PadLen);

            for (uint16_t n = 0; n < dimNNum; n++) {
                for (uint16_t c = 0; c < dimCNum; c++) {
                    uint32_t tempOffset = startOffset + c * padHW + n * padCHW;
                    for (uint16_t i = 0; i < padVLNum; i++) {
                        AscendC::Reg::DataCopy(vRegTmp, dstAddr + tempOffset + InOffset + i * VL_ELEMS);
                        AscendC::Reg::DataCopy(dstAddr + tempOffset + OutOffset + i * VL_ELEMS, vRegTmp, maskAll);
                    }
                    for (uint16_t i = 0; i < BLNum; i++) {
                        AscendC::Reg::DataCopy(vRegTmp, dstAddr + tempOffset + InOffset + padVLNum * VL_ELEMS);
                        AscendC::Reg::DataCopy(dstAddr + tempOffset + OutOffset + padVLNum * VL_ELEMS, vRegTmp,
                                               endMask);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& dst, PadCircNormalParam& padParam, int32_t idx)
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
            int64_t upIndex = calOutIndex(i, true);
            int64_t downIndex = calOutIndex(i, false);
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

        for (auto i = curFromIndex; i < curToIndex; i++) {
            CopyOutOne(dst, padParam, Indexes[i]);
        }
        if (!lastOut_) {
            if ((idx & 1) == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void CopyOutOne(const LocalTensor<T>& dst, PadCircNormalParam& padParam, int64_t indexOffset)
    {
        const int8_t dimNum = tilingData_->dimNum;
        const int8_t ubAxis = tilingData_->ubAxis;
        uint64_t firstLeftPad = tilingData_->leftPad[dimNum - 1];
        uint64_t addAddr = 0;
        if (upOffset_ > 0) {
            addAddr = additionOffset_ / HALF_SIZE;
        }
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = tilingData_->outShape[dimNum - 1] * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyExtParams copyOutParamsF;
        copyOutParamsF.blockCount = 1;
        copyOutParamsF.blockLen = firstLeftPad * sizeof(T);
        copyOutParamsF.srcStride = 0;
        copyOutParamsF.dstStride = 0;
        DataCopyExtParams copyOutParamsL;
        copyOutParamsL.blockCount = 1;
        copyOutParamsL.blockLen = (tilingData_->outShape[dimNum - 1] - firstLeftPad) * sizeof(T);
        copyOutParamsL.srcStride = 0;
        copyOutParamsL.dstStride = 0;
        uint64_t outOffset = outIndex_[ubAxis] * tilingData_->outStride[ubAxis];

        DataCopyPad(output_[indexOffset + outOffset], dst, copyOutParamsF);
        copyOutParams.blockCount = blockCount_ - 1;
        DataCopyPad(output_[indexOffset + outOffset + firstLeftPad], dst[additionOffset_], copyOutParams);
        DataCopyPad(output_[indexOffset + outOffset + firstLeftPad +
                            copyOutParams.blockCount * tilingData_->outShape[dimNum - 1]],
                    dst[additionOffset_ + copyOutParams.blockCount * padParam.padWO], copyOutParamsL);

        if (blockCountU_ > 0) {
            uint64_t outOffsetU = inIndex_[ubAxis] <= tilingData_->inShape[ubAxis] - tilingData_->leftPad[ubAxis] ?
                                      0 :
                                      (outIndex_[ubAxis] - tilingData_->inShape[ubAxis]) *
                                          tilingData_->outStride[ubAxis];
            DataCopyPad(output_[indexOffset + outOffsetU], dst[addAddr], copyOutParamsF);
            copyOutParams.blockCount = blockCountU_ - 1;
            DataCopyPad(output_[indexOffset + outOffsetU + firstLeftPad], dst[additionOffset_ + upOffset_],
                        copyOutParams);
            DataCopyPad(output_[indexOffset + outOffsetU + firstLeftPad +
                                copyOutParams.blockCount * tilingData_->outShape[dimNum - 1]],
                        dst[additionOffset_ + copyOutParams.blockCount * padParam.padWO + upOffset_], copyOutParamsL);
        }
        if (blockCountD_ > 0) {
            uint64_t outOffsetD = (outIndex_[ubAxis] + tilingData_->inShape[ubAxis]) * tilingData_->outStride[ubAxis];
            DataCopyPad(output_[indexOffset + outOffsetD], dst, copyOutParamsF);
            copyOutParams.blockCount = blockCountD_ - 1;
            DataCopyPad(output_[indexOffset + outOffsetD + firstLeftPad], dst[additionOffset_], copyOutParams);
            DataCopyPad(output_[indexOffset + outOffsetD + firstLeftPad +
                                copyOutParams.blockCount * tilingData_->outShape[dimNum - 1]],
                        dst[additionOffset_ + copyOutParams.blockCount * padParam.padWO], copyOutParamsL);
        }
    }

    __aicore__ inline int64_t calOutIndex(int8_t curAxis, bool isUp)
    {
        int64_t res = -1;
        if (isUp) {
            uint64_t first = tilingData_->inShape[curAxis] - tilingData_->leftPad[curAxis];
            uint64_t last = tilingData_->inShape[curAxis];
            if (inIndex_[curAxis] < last && inIndex_[curAxis] >= first) {
                res = inIndex_[curAxis] - first;
            }
        } else {
            uint64_t first = 0;
            uint64_t last = tilingData_->outShape[curAxis] - tilingData_->inShape[curAxis] -
                            tilingData_->leftPad[curAxis];
            if (inIndex_[curAxis] < last && inIndex_[curAxis] >= first) {
                res = tilingData_->leftPad[curAxis] + tilingData_->inShape[curAxis] + inIndex_[curAxis];
            }
        }
        return res;
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
