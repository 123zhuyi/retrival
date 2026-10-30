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
 * \file pad_mirror_gather.h
 * \brief pad gather kernel
 */

#ifndef ASCENDC_PAD_MIRROR_GATHER_H_
#define ASCENDC_PAD_MIRROR_GATHER_H_

#include "kernel_operator.h"
#include "pad_v3_struct.h"
#include "pad_common.h"

namespace PadV3 {
using namespace AscendC;

template <typename T, int32_t KEY>
class PadMirrorGather {
private:
    using RangeType = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType = std::conditional_t<sizeof(T) == 1,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    using OutType = std::conditional_t<sizeof(T) != sizeof(uint64_t), T, uint32_t>;
    constexpr static uint32_t VL_RANGE_CNT = VL_SIZE / sizeof(RangeType);
    constexpr static uint32_t UB_AXES = (KEY / KEY_BASE) % KEY_BASE; // TilingKey倒数第二维为UB内轴个数
    constexpr static bool IS_REFLECT = (KEY / 1000) % KEY_BASE == 1; // TilingKey倒数第四维标识reflect还是symmetric
    constexpr static uint32_t VL_CNT = VL_SIZE / sizeof(T);
    constexpr static uint32_t BLOCK_NUM = UB_BLOCK / sizeof(T);
    constexpr static uint32_t MAX_DIM = 8;
    constexpr static int32_t BUF_NUM = 2; // double buffer
    constexpr static int32_t CONST2 = 2;
    constexpr static int32_t CONST3 = 3;
    constexpr static int32_t CONST4 = 4;

private:
    TPipe* pipe_ = nullptr;
    const PadACTilingData* tdPtr_ = nullptr;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQueFw_; // ub切分轴对应的原始输出
    TQue<QuePosition::VECOUT, 1> outQueBw_; // ub切分轴对应的pad
    TBuf<QuePosition::VECCALC> idxBuf_;

    int32_t blockIdx_{0};
    int32_t dimNum_{0};
    int32_t ubAxis_{0};
    int32_t ubFactor_{0};
    uint16_t vlSplitIn_{0};        // VL切分轴的factor
    bool lastThirdDimInVL_{false}; // 一次VL是否处理后面三根轴

    struct OutIndicesSet {
        uint64_t outIdx[CONST3];
        int32_t count = 0;
    };

public:
    __aicore__ inline PadMirrorGather(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, const PadACTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tdPtr_ = tilingData;
        dimNum_ = tdPtr_->dimNum;
        ubAxis_ = tdPtr_->ubAxis;
        ubFactor_ = tdPtr_->ubFactor;

        // 一次VL需要处理后面3根轴，当前支持切-3轴、-4轴
        if (dimNum_ >= CONST3 && tdPtr_->outStride[dimNum_ - CONST3] <= VL_CNT / CONST2) {
            lastThirdDimInVL_ = true;
            vlSplitIn_ = Std::min(uint64_t(VL_CNT / tdPtr_->outStride[dimNum_ - CONST3]),
                                  uint64_t(tdPtr_->outShape[dimNum_ - CONST3]));
        } else {
            vlSplitIn_ = Std::min(uint64_t(VL_CNT / tdPtr_->outStride[dimNum_ - CONST2]),
                                  uint64_t(tdPtr_->outShape[dimNum_ - CONST2]));
        }
        if constexpr (sizeof(T) == 1) {
            vlSplitIn_ /= sizeof(int16_t);
            if (vlSplitIn_ == 0) {
                vlSplitIn_ = 1;
            }
        }

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBuffer(inQue_, BUF_NUM, tdPtr_->outTileSize);
        // 正/反向各一份outTileSize; 反向上/下pad各一个blocksize的临时空间
        pipe_->InitBuffer(outQueFw_, BUF_NUM, tdPtr_->outTileSize);
        pipe_->InitBuffer(outQueBw_, BUF_NUM, tdPtr_->outTileSize + UB_BLOCK * CONST2);
        pipe_->InitBuffer(idxBuf_, VL_SIZE * CONST2); // 正向索引占前VL; 反向索引占后VL
    }

    __aicore__ inline void Process()
    {
        uint32_t startIdx = blockIdx_ * tdPtr_->ubPerCount;
        if (startIdx >= tdPtr_->ubTotalCount) {
            return;
        }

        uint32_t endIdx = (blockIdx_ + 1L) * tdPtr_->ubPerCount;
        endIdx = endIdx < tdPtr_->ubTotalCount ? endIdx : tdPtr_->ubTotalCount;

        LocalTensor<RangeType> idxTensor = idxBuf_.Get<RangeType>();
        if (lastThirdDimInVL_) {
            GenGatherIndexThreeDim(idxTensor);
        } else {
            GenGatherIndex(idxTensor);
        }

        uint64_t ubTailFactor = tdPtr_->inShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint64_t inIndex[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
            uint16_t ubAxisInCopyNum = 0;

            CalcDimIdx(idx, inIndex);
            ubAxisInCopyNum = (inIndex[ubAxis_] + ubFactor_ > tdPtr_->inShape[ubAxis_]) ? ubTailFactor : ubFactor_;

            ProcessOneStep(inIndex, ubAxisInCopyNum, idxTensor);
        }
    }

private:
    __aicore__ inline void CalcDimIdx(uint32_t curIdx, uint64_t* inIndex)
    {
        for (int32_t i = ubAxis_; i >= 0; i--) {
            uint64_t factor = tdPtr_->inShape[i];
            if (i == ubAxis_) {
                factor = CeilDiv(factor, static_cast<uint64_t>(ubFactor_));
            }
            inIndex[i] = (i == ubAxis_ ? curIdx % factor * ubFactor_ : curIdx % factor);
            curIdx = curIdx / factor;
        }
    }

    __aicore__ inline void ProcessOneStep(const uint64_t* inIndex, uint16_t ubAxisInCopyNum,
                                          LocalTensor<RangeType>& idxTensor)
    {
        CopyIn(inIndex, ubAxisInCopyNum);
        GatherAndCopyOut(ubAxisInCopyNum, inIndex, idxTensor);
    }

    __aicore__ inline void CopyIn(const uint64_t* inIndex, uint16_t ubAxisInCopyNum)
    {
        uint32_t copyInNum = ubAxisInCopyNum * tdPtr_->inStride[ubAxis_];
        uint64_t inAddr = 0;
        for (uint32_t i = 0; i < dimNum_; i++) {
            inAddr += inIndex[i] * tdPtr_->inStride[i];
        }

        LocalTensor<T> inLocal = inQue_.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams = {1u, static_cast<uint32_t>(copyInNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(inLocal, inputGm_[inAddr], copyInParams, padParams);

        inQue_.EnQue(inLocal);
    }

    __aicore__ inline void GatherAndCopyOut(uint16_t ubAxisInCopyNum, const uint64_t* inIndex,
                                            LocalTensor<RangeType>& idxTensor)
    {
        // 最大支持5维input, ub内至少2根轴，所以最多4根轴
        OutIndicesSet totalOutIdx[CONST4];
        CalOutIndexValue(inIndex, totalOutIdx, CONST4);

        LocalTensor<T> inLocal = inQue_.DeQue<T>();

        LocalTensor<T> outLocalFw = outQueFw_.AllocTensor<T>();
        if constexpr (UB_AXES == CONST2) {
            GatherProcessUb2DFw(idxTensor, inLocal, outLocalFw, ubAxisInCopyNum);
        } else if constexpr (UB_AXES == CONST3) {
            GatherProcessUb3DFw(idxTensor, inLocal, outLocalFw, ubAxisInCopyNum);
        } else if constexpr (UB_AXES == CONST4) {
            GatherProcessUb4DFw(idxTensor, inLocal, outLocalFw, ubAxisInCopyNum);
        }

        outQueFw_.EnQue(outLocalFw);
        CopyOutFw(ubAxisInCopyNum, inIndex, totalOutIdx);

        LocalTensor<T> outLocalBw = outQueBw_.AllocTensor<T>();
        if constexpr (UB_AXES == CONST2) {
            if (tdPtr_->inShape[dimNum_ - CONST2] != tdPtr_->outShape[dimNum_ - CONST2]) {
                GatherProcessUb2DBw(idxTensor, inLocal, outLocalBw, ubAxisInCopyNum);
            }
        } else if constexpr (UB_AXES == CONST3) {
            if (tdPtr_->inShape[dimNum_ - CONST3] != tdPtr_->outShape[dimNum_ - CONST3]) {
                GatherProcessUb3DBw(idxTensor, inLocal, outLocalBw, ubAxisInCopyNum);
            }
        } else if constexpr (UB_AXES == CONST4) {
            if (tdPtr_->inShape[dimNum_ - CONST4] != tdPtr_->outShape[dimNum_ - CONST4]) {
                GatherProcessUb4DBw(idxTensor, inLocal, outLocalBw, ubAxisInCopyNum);
            }
        }

        outQueBw_.EnQue(outLocalBw);
        inQue_.FreeTensor(inLocal);

        CopyOutBw(ubAxisInCopyNum, inIndex, totalOutIdx);
    }

    __aicore__ inline bool IsInLeftPad(int64_t inIdx, int32_t inAxis)
    {
        if constexpr (IS_REFLECT) {
            return inIdx > 0 && inIdx <= (int64_t)tdPtr_->leftPad[inAxis];
        } else {
            return inIdx >= 0 && inIdx < (int64_t)tdPtr_->leftPad[inAxis];
        }
    }

    __aicore__ inline bool IsInRightPad(int64_t inIdx, int32_t inAxis)
    {
        // 右pad  leftNum+inShape-1 < outIdx < outShape
        if constexpr (IS_REFLECT) {
            return (inIdx < (int64_t)tdPtr_->inShape[inAxis] - 1 &&
                    inIdx > (int64_t)tdPtr_->leftPad[inAxis] + CONST2 * ((int64_t)tdPtr_->inShape[inAxis] - 1) -
                                (int64_t)tdPtr_->outShape[inAxis]);
        } else {
            return (inIdx < (int64_t)tdPtr_->inShape[inAxis] && inIdx > (int64_t)tdPtr_->leftPad[inAxis] +
                                                                            CONST2 * (int64_t)tdPtr_->inShape[inAxis] -
                                                                            1 - (int64_t)tdPtr_->outShape[inAxis]);
        }
    }

    __aicore__ inline void CalOutIndexValue(const uint64_t* inIndex, OutIndicesSet* totalOutIdx, int32_t setMax)
    {
        /***************************
        reflect:
            原始位置: outIdx=leftNum+inIdx
            左pad:   outIdx=leftNum-inIdx
            右pad:   outIdx=leftNum+inShape-1+(inShape-1-inIdx)=leftNum+2*(inshape-1)-inIdx
        symmetric:
            原始位置: outIdx=leftNum+inIdx
            左pad:   outIdx=leftNum-1-inIdx
            右pad:   outIdx=leftNum+inShape-1+(inShape-inIdx)=leftNum+2*inshape-1-inIdx
        ***************************/

        // ub切分轴左侧，左/右pad以及原始的正向
        for (int32_t i = 0; i < ubAxis_; i++) {
            uint64_t iDimInIdx = inIndex[i];
            if constexpr (IS_REFLECT) {
                // 左pad
                if (IsInLeftPad(iDimInIdx, i)) {
                    totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] - iDimInIdx;
                    totalOutIdx[i].count++;
                }
                // 原始
                totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] + iDimInIdx;
                totalOutIdx[i].count++;
                // 右pad  leftNum+inShape-1 < outIdx < outShape
                if (IsInRightPad(iDimInIdx, i)) {
                    totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] +
                                                                  CONST2 * (tdPtr_->inShape[i] - 1) - iDimInIdx;
                    totalOutIdx[i].count++;
                }
            } else {
                // 左pad
                if (IsInLeftPad(iDimInIdx, i)) {
                    totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] - 1 - iDimInIdx;
                    totalOutIdx[i].count++;
                }
                // 原始
                totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] + iDimInIdx;
                totalOutIdx[i].count++;
                // 右pad  leftNum+inShape-1 < outIdx < outShape
                if (IsInRightPad(iDimInIdx, i)) {
                    totalOutIdx[i].outIdx[totalOutIdx[i].count] = tdPtr_->leftPad[i] + CONST2 * tdPtr_->inShape[i] - 1 -
                                                                  iDimInIdx;
                    totalOutIdx[i].count++;
                }
            }
        }

        // ub切分轴，正反向单独处理，此处不赋值

        // ub切分轴右侧，已经在ub中补好了所有的pad，此处置0
        for (int32_t i = ubAxis_ + 1; i < setMax; i++) {
            totalOutIdx[i].outIdx[0] = 0;
            totalOutIdx[i].count = 1;
        }
    }

    __aicore__ inline void CopyOutFw(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, OutIndicesSet* totalOutIdx)
    {
        LocalTensor<T> outLocalFw = outQueFw_.DeQue<T>();

        uint32_t copyOutNum = ubAxisInCopyNum * tdPtr_->outStride[ubAxis_];
        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNum * sizeof(T)), 0, 0, 0};

        // ub切分轴，正向
        totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] + inIndex[ubAxis_];
        totalOutIdx[ubAxis_].count = 1;

        for (int32_t o0 = 0; o0 < totalOutIdx[0].count; o0++) {
            uint64_t o0Offset = totalOutIdx[0].outIdx[o0] * tdPtr_->outStride[0];
            for (int32_t o1 = 0; o1 < totalOutIdx[1].count; o1++) {
                uint64_t o1Offset = totalOutIdx[1].outIdx[o1] * tdPtr_->outStride[1];
                for (int32_t o2 = 0; o2 < totalOutIdx[CONST2].count; o2++) {
                    uint64_t o2Offset = totalOutIdx[CONST2].outIdx[o2] * tdPtr_->outStride[CONST2];
                    for (int32_t o3 = 0; o3 < totalOutIdx[CONST3].count; o3++) {
                        uint64_t o3Offset = totalOutIdx[CONST3].outIdx[o3] * tdPtr_->outStride[CONST3];
                        uint64_t outAddr = o0Offset + o1Offset + o2Offset + o3Offset;
                        DataCopyPad(outputGm_[outAddr], outLocalFw, copyOutParams);
                    }
                }
            }
        }

        outQueFw_.FreeTensor(outLocalFw);
    }

    __aicore__ inline void CalcUbAxisPadBwNum(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, uint32_t& inLeftPadNum,
                                              uint32_t& inLeftPadStart, uint32_t& inRightPadNum,
                                              uint32_t& inRightPadStart)
    {
        if constexpr (IS_REFLECT) {
            if ((inIndex[ubAxis_] == 0 && ubAxisInCopyNum > 1 && tdPtr_->leftPad[ubAxis_] > 0)) {
                inLeftPadNum = Std::min((uint64_t)ubAxisInCopyNum - 1, (uint64_t)tdPtr_->leftPad[ubAxis_]);
                inLeftPadStart = ubAxisInCopyNum - 1 - inLeftPadNum;
            } else if (IsInLeftPad(inIndex[ubAxis_], ubAxis_)) {
                inLeftPadNum = Std::min((uint64_t)ubAxisInCopyNum,
                                        (uint64_t)tdPtr_->leftPad[ubAxis_] - inIndex[ubAxis_] + 1);
                inLeftPadStart = ubAxisInCopyNum - inLeftPadNum;
            }
        } else {
            if (IsInLeftPad(inIndex[ubAxis_], ubAxis_)) {
                inLeftPadNum = Std::min((uint64_t)ubAxisInCopyNum,
                                        (uint64_t)tdPtr_->leftPad[ubAxis_] - inIndex[ubAxis_]);
                inLeftPadStart = ubAxisInCopyNum - inLeftPadNum;
            }
        }

        uint64_t rightIdx = inIndex[ubAxis_] + ubAxisInCopyNum - 1;
        int64_t rightPad = tdPtr_->outShape[ubAxis_] - tdPtr_->inShape[ubAxis_] - tdPtr_->leftPad[ubAxis_];
        // reflect: outIdx=leftNum+2*(inshape-1)-inIdx=outshape-1
        // symmetric: outIdx=leftNum+2*inshape-1-inIdx=outshape-1
        // 当outIdx取最大值时，可以得到对应的最小的inIdx
        if constexpr (IS_REFLECT) {
            uint64_t minInIdx = tdPtr_->leftPad[ubAxis_] + 2 * (tdPtr_->inShape[ubAxis_] - 1) -
                                (tdPtr_->outShape[ubAxis_] - 1);
            if (rightIdx == tdPtr_->inShape[ubAxis_] - 1 && ubAxisInCopyNum > 1 && rightPad > 0) {
                inRightPadNum = Std::min((uint64_t)ubAxisInCopyNum - 1, (uint64_t)rightIdx - minInIdx);
                inRightPadStart = 1;
            } else if (IsInRightPad(rightIdx, ubAxis_)) {
                inRightPadNum = Std::min((uint64_t)ubAxisInCopyNum, (uint64_t)rightIdx - minInIdx + 1);
                inRightPadStart = 0;
            }
        } else {
            if (IsInRightPad(rightIdx, ubAxis_)) {
                uint64_t minInIdx = tdPtr_->leftPad[ubAxis_] + 2 * tdPtr_->inShape[ubAxis_] - tdPtr_->outShape[ubAxis_];
                inRightPadNum = Std::min((uint64_t)ubAxisInCopyNum, (uint64_t)rightIdx - minInIdx + 1);
                inRightPadStart = 0;
            }
        }
    }

    __aicore__ inline void CopyOutPadBwInnerLoop(LocalTensor<T>& outLocalBwReal, LocalTensor<T>& outLocalBwTmp,
                                                 uint32_t copyStartOffset, uint32_t copyOutNum, uint32_t alignOffset,
                                                 DataCopyExtParams& copyOutParams, DataCopyExtParams& outParamAlign,
                                                 OutIndicesSet* totalOutIdx, uint64_t baseOffset)
    {
        for (int32_t o3 = 0; o3 < totalOutIdx[CONST3].count; o3++) {
            uint64_t outAddr = baseOffset + totalOutIdx[CONST3].outIdx[o3] * tdPtr_->outStride[CONST3];
            if (copyOutNum > 0) {
                DataCopyPad(outputGm_[outAddr + alignOffset], outLocalBwReal[copyStartOffset], copyOutParams);
            }
            if (alignOffset > 0) {
                DataCopyPad(outputGm_[outAddr], outLocalBwTmp, outParamAlign);
            }
        }
    }

    __aicore__ inline void CopyOutPadBwLoopImpl(LocalTensor<T>& outLocalBwReal, LocalTensor<T>& outLocalBwTmp,
                                                uint32_t copyStartOffset, uint32_t copyOutNum, uint32_t alignOffset,
                                                DataCopyExtParams& copyOutParams, DataCopyExtParams& outParamAlign,
                                                OutIndicesSet* totalOutIdx)
    {
        for (int32_t o0 = 0; o0 < totalOutIdx[0].count; o0++) {
            uint64_t o0Offset = totalOutIdx[0].outIdx[o0] * tdPtr_->outStride[0];
            for (int32_t o1 = 0; o1 < totalOutIdx[1].count; o1++) {
                uint64_t o1Offset = totalOutIdx[1].outIdx[o1] * tdPtr_->outStride[1];
                for (int32_t o2 = 0; o2 < totalOutIdx[CONST2].count; o2++) {
                    uint64_t baseOffset = o0Offset + o1Offset +
                                          totalOutIdx[CONST2].outIdx[o2] * tdPtr_->outStride[CONST2];
                    CopyOutPadBwInnerLoop(outLocalBwReal, outLocalBwTmp, copyStartOffset, copyOutNum, alignOffset,
                                          copyOutParams, outParamAlign, totalOutIdx, baseOffset);
                }
            }
        }
    }

    __aicore__ inline void CopyOutPadBwImpl(LocalTensor<T>& outLocalBwReal, LocalTensor<T>& outLocalBwTmp,
                                            uint32_t copyStartOffset, uint32_t copyOutNum, OutIndicesSet* totalOutIdx)
    {
        uint32_t alignRed = copyStartOffset % BLOCK_NUM;
        uint32_t alignOffset = 0;
        if (alignRed != 0) {
            __local_mem__ T* inAddrTmp = (__local_mem__ T*)outLocalBwReal.GetPhyAddr() + copyStartOffset;
            __local_mem__ T* outAddrTmp = (__local_mem__ T*)outLocalBwTmp.GetPhyAddr();

            alignOffset = BLOCK_NUM - alignRed;
            copyStartOffset = copyStartOffset + alignOffset;
            if (copyOutNum > alignOffset) {
                copyOutNum = (copyOutNum - alignOffset);
            } else {
                alignOffset = copyOutNum;
                copyOutNum = 0;
            }

            CopyTmpUnAlign(inAddrTmp, outAddrTmp, alignOffset);

            SetWaitEvent<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        }

        DataCopyExtParams copyOutParams = {1u, static_cast<uint32_t>(copyOutNum * sizeof(T)), 0, 0, 0};
        DataCopyExtParams outParamAlign = {1u, static_cast<uint32_t>(alignOffset * sizeof(T)), 0, 0, 0};

        CopyOutPadBwLoopImpl(outLocalBwReal, outLocalBwTmp, copyStartOffset, copyOutNum, alignOffset, copyOutParams,
                             outParamAlign, totalOutIdx);
    }

    __aicore__ inline void CopyOutLeftPadBw(uint16_t ubAxisInCopyNum, const uint64_t* inIndex,
                                            LocalTensor<T>& outLocalBw, uint32_t inLeftPadNum, uint32_t inLeftPadStart,
                                            OutIndicesSet* totalOutIdx)
    {
        // reflect:
        //     左pad:   outIdx=leftNum-inIdx
        // symmetric:
        //     左pad:   outIdx=leftNum-1-inIdx

        // ub切分轴，leftpad
        uint64_t inIdx = inIndex[ubAxis_] + ubAxisInCopyNum - 1 - inLeftPadStart;
        if constexpr (IS_REFLECT) {
            totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] - inIdx;
        } else {
            totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] - 1 - inIdx;
        }
        totalOutIdx[ubAxis_].count = 1;

        LocalTensor<T> outLocalBwReal = outLocalBw[BLOCK_NUM * CONST2];
        LocalTensor<T> outLocalBwTmp = outLocalBw[0];
        uint32_t copyOutNum = inLeftPadNum * tdPtr_->outStride[ubAxis_];
        uint32_t copyStartOffset = inLeftPadStart * tdPtr_->outStride[ubAxis_];

        CopyOutPadBwImpl(outLocalBwReal, outLocalBwTmp, copyStartOffset, copyOutNum, totalOutIdx);
    }

    __aicore__ inline void CopyOutRightPadBw(uint16_t ubAxisInCopyNum, const uint64_t* inIndex,
                                             LocalTensor<T>& outLocalBw, uint32_t inRightPadNum,
                                             uint32_t inRightPadStart, OutIndicesSet* totalOutIdx)
    {
        // reflect:
        //     右pad:   outIdx=leftNum+inShape-1+(inShape-1-inIdx)=leftNum+2*(inshape-1)-inIdx
        // symmetric:
        //     右pad:   outIdx=leftNum+inShape-1+(inShape-inIdx)=leftNum+2*inshape-1-inIdx

        // ub切分轴，rightpad
        uint64_t inIdx = inIndex[ubAxis_] + ubAxisInCopyNum - 1 - inRightPadStart;
        if constexpr (IS_REFLECT) {
            totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] + CONST2 * (tdPtr_->inShape[ubAxis_] - 1) - inIdx;
        } else {
            totalOutIdx[ubAxis_].outIdx[0] = tdPtr_->leftPad[ubAxis_] + CONST2 * tdPtr_->inShape[ubAxis_] - 1 - inIdx;
        }
        totalOutIdx[ubAxis_].count = 1;

        LocalTensor<T> outLocalBwReal = outLocalBw[BLOCK_NUM * CONST2];
        LocalTensor<T> outLocalBwTmp = outLocalBw[BLOCK_NUM];
        uint32_t copyOutNum = inRightPadNum * tdPtr_->outStride[ubAxis_];
        uint32_t copyStartOffset = inRightPadStart * tdPtr_->outStride[ubAxis_];

        CopyOutPadBwImpl(outLocalBwReal, outLocalBwTmp, copyStartOffset, copyOutNum, totalOutIdx);
    }

    __aicore__ inline void CopyTmpUnAlign(__local_mem__ T* inAddrTmp, __local_mem__ T* outAddrTmp, uint32_t alignOffset)
    {
        uint32_t newCnt = (sizeof(T) != sizeof(OutType)) ? CONST2 * alignOffset : alignOffset;
        auto newInAddr = reinterpret_cast<__local_mem__ OutType*>(inAddrTmp);
        auto newOutAddr = reinterpret_cast<__local_mem__ OutType*>(outAddrTmp);

        __VEC_SCOPE__
        {
            Reg::RegTensor<OutType> regData;
            Reg::UnalignReg uReg;
            Reg::MaskReg maskIdx = Reg::UpdateMask<OutType>(newCnt);

            Reg::DataCopyUnAlignPre(uReg, newInAddr);
            Reg::DataCopyUnAlign(regData, uReg, newInAddr, newCnt);
            Reg::DataCopy(newOutAddr, regData, maskIdx);
        }
    }

    __aicore__ inline void CopyOutBw(uint16_t ubAxisInCopyNum, const uint64_t* inIndex, OutIndicesSet* totalOutIdx)
    {
        LocalTensor<T> outLocalBw = outQueBw_.DeQue<T>();

        // 计算当前ub切分轴上，该次搬运的inputGm是否在左、右pad上，以及左右pad的起始位置
        uint32_t inLeftPadNum = 0;
        uint32_t inLeftPadStart = 0;
        uint32_t inRightPadNum = 0;
        uint32_t inRightPadStart = 0;

        CalcUbAxisPadBwNum(ubAxisInCopyNum, inIndex, inLeftPadNum, inLeftPadStart, inRightPadNum, inRightPadStart);

        if (inLeftPadNum != 0) {
            CopyOutLeftPadBw(ubAxisInCopyNum, inIndex, outLocalBw, inLeftPadNum, inLeftPadStart, totalOutIdx);
        }

        if (inRightPadNum != 0) {
            CopyOutRightPadBw(ubAxisInCopyNum, inIndex, outLocalBw, inRightPadNum, inRightPadStart, totalOutIdx);
        }

        outQueBw_.FreeTensor(outLocalBw);
    }

    __aicore__ inline void GenGatherIndexThreeDim(LocalTensor<RangeType>& idxTensor)
    {
        // 2*VL长度的索引，纯PAD索引占前VL，正常值索引占后VL
        uint32_t lastInDimSize = tdPtr_->inShape[dimNum_ - 1];
        uint16_t lastSecInDimSize = tdPtr_->inShape[dimNum_ - CONST2];
        int32_t outStride1 = tdPtr_->outStride[dimNum_ - CONST3];
        int32_t outStride2 = tdPtr_->outStride[dimNum_ - CONST2];
        int32_t inStride1 = tdPtr_->inStride[dimNum_ - CONST3];
        uint16_t lastTwoDimLoops = vlSplitIn_;
        // 切在-3轴上，-2轴上的数据都是从gather中获取到的，包含pad
        int32_t lastLeftPadNum = tdPtr_->leftPad[dimNum_ - 1];
        uint16_t last2LeftPadNum = tdPtr_->leftPad[dimNum_ - CONST2];
        uint16_t last2RightPadNum = tdPtr_->outShape[dimNum_ - CONST2] - lastSecInDimSize - last2LeftPadNum;
        RangeType lastDimIdx = lastInDimSize - 1;
        RangeType last2DimIdx = lastSecInDimSize - 1;
        uint32_t decreaseOffset = VL_RANGE_CNT - 1;
        uint32_t lastRightPadArangeStart = lastLeftPadNum + 2 * lastDimIdx;
        uint32_t modeOffset = 1;
        if constexpr (IS_REFLECT) {
            modeOffset = 0;
        }

        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr2 = (__local_mem__ RangeType*)idxTensor.GetPhyAddr() + VL_RANGE_CNT;

        __VEC_SCOPE__
        {
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> lineRange;
            Reg::RegTensor<RangeType> lineRangeNew;
            Reg::RegTensor<RangeType> lineRangeBk;
            Reg::MaskReg leftMask;
            Reg::RegTensor<RangeType> leftPadIdxReg;
            Reg::MaskReg rightMask;
            Reg::RegTensor<RangeType> rightPadIdxReg;
            Reg::UnalignReg uReg;
            Reg::UnalignReg uRegIn;

            // 先拼好-1轴的索引
            Reg::Arange(lineRange, 0);
            Reg::Adds(lineRange, lineRange, ((RangeType)-1) * lastLeftPadNum, maskIdx);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(leftMask, lineRange, 0, maskIdx);
            Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(leftPadIdxReg,
                                                                    lastLeftPadNum - decreaseOffset - modeOffset);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, leftPadIdxReg, leftMask);

            Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(
                rightPadIdxReg, lastRightPadArangeStart - decreaseOffset + modeOffset);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(rightMask, rightPadIdxReg, lastDimIdx + modeOffset, maskIdx);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, rightPadIdxReg, rightMask);

            // -3轴纯pad的索引, 包含末尾两根轴
            for (uint16_t i = 0; i < lastTwoDimLoops; i++) {
                Reg::Adds(lineRangeBk, lineRange, (RangeType)((lastTwoDimLoops - i - 1) * inStride1), maskIdx);
                // -2轴leftpad行数, 索引都一样
                __local_mem__ RangeType* idxAddrTmp = idxAddr + i * outStride1;
                for (uint16_t j = 0; j < last2LeftPadNum; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk,
                              (RangeType)((last2LeftPadNum - j - modeOffset) * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

                // -2 轴inputshape, 索引递增
                __local_mem__ RangeType* idxAddrTmp1 = idxAddr + i * outStride1 + last2LeftPadNum * outStride2;
                for (uint16_t j = 0; j < lastSecInDimSize; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(j * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp1, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp1, uReg, 0);

                // -2轴rightpad行数, 索引都一样
                Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(last2DimIdx * lastInDimSize), maskIdx);
                __local_mem__ RangeType* idxAddrTmp2 = idxAddr + i * outStride1 + last2LeftPadNum * outStride2 +
                                                       lastSecInDimSize * outStride2;
                for (uint16_t j = 0; j < last2RightPadNum; j++) {
                    Reg::Adds(lineRangeBk, lineRangeNew, (RangeType)((-1) * (j + 1 - modeOffset) * lastInDimSize),
                              maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeBk, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp2, uReg, 0);
            }

            // -3轴有效输入的索引
            for (uint16_t i = 0; i < lastTwoDimLoops; i++) {
                Reg::Adds(lineRangeBk, lineRange, (RangeType)(i * inStride1), maskIdx);
                // -2轴leftpad行数, 索引都一样
                __local_mem__ RangeType* idxAddrTmp = idxAddr2 + i * outStride1;
                for (uint16_t j = 0; j < last2LeftPadNum; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk,
                              (RangeType)((last2LeftPadNum - j - modeOffset) * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

                // -2 轴inputshape, 索引递增
                __local_mem__ RangeType* idxAddrTmp1 = idxAddr2 + i * outStride1 + last2LeftPadNum * outStride2;
                for (uint16_t j = 0; j < lastSecInDimSize; j++) {
                    Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(j * lastInDimSize), maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp1, lineRangeNew, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp1, uReg, 0);

                // -2轴rightpad行数, 索引都一样
                Reg::Adds(lineRangeNew, lineRangeBk, (RangeType)(last2DimIdx * lastInDimSize), maskIdx);
                __local_mem__ RangeType* idxAddrTmp2 = idxAddr2 + i * outStride1 + last2LeftPadNum * outStride2 +
                                                       lastSecInDimSize * outStride2;
                for (uint16_t j = 0; j < last2RightPadNum; j++) {
                    Reg::Adds(lineRangeBk, lineRangeNew, (RangeType)((-1) * (j + 1 - modeOffset) * lastInDimSize),
                              maskIdx);
                    Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeBk, uReg, outStride2);
                }
                Reg::DataCopyUnAlignPost(idxAddrTmp2, uReg, 0);
            }
        }
    }

    __aicore__ inline void GenGatherIndex(LocalTensor<RangeType>& idxTensor)
    {
        // 2*VL长度的索引，纯PAD索引占前VL，正常值索引占后VL
        uint32_t lastInDimSize = tdPtr_->inShape[dimNum_ - 1];
        int32_t lastLeftPadNum = tdPtr_->leftPad[dimNum_ - 1];
        uint32_t lastOutDimSize = tdPtr_->outShape[dimNum_ - 1];
        uint16_t lastDimsLeft = vlSplitIn_;
        RangeType lastDimIdx = lastInDimSize - 1;
        __local_mem__ RangeType* idxAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr2 = (__local_mem__ RangeType*)idxTensor.GetPhyAddr() + VL_RANGE_CNT;
        uint32_t decreaseOffset = VL_RANGE_CNT - 1;
        uint32_t lastRightPadArangeStart = lastLeftPadNum + 2 * lastDimIdx;
        uint32_t modeOffset = 1;
        if constexpr (IS_REFLECT) {
            modeOffset = 0;
        }

        __VEC_SCOPE__
        {
            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::RegTensor<RangeType> lineRange;
            Reg::RegTensor<RangeType> lineRangeNew;
            Reg::MaskReg leftMask;
            Reg::RegTensor<RangeType> leftPadIdxReg;
            Reg::MaskReg rightMask;
            Reg::RegTensor<RangeType> rightPadIdxReg;
            Reg::UnalignReg uReg;
            Reg::UnalignReg uRegIn;

            // 先拼好-1轴的索引
            Reg::Arange(lineRange, 0);
            // 先拷出去，防止索引尾部脏数据
            Reg::DataCopy(idxAddr, lineRange, maskIdx);
            Reg::DataCopy(idxAddr2, lineRange, maskIdx);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

            Reg::Adds(lineRange, lineRange, ((RangeType)-1) * lastLeftPadNum, maskIdx);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(leftMask, lineRange, 0, maskIdx);
            Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(leftPadIdxReg,
                                                                    lastLeftPadNum - decreaseOffset - modeOffset);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, leftPadIdxReg, leftMask);

            Reg::Arange<RangeType, Reg::IndexOrder::DECREASE_ORDER>(
                rightPadIdxReg, lastRightPadArangeStart - decreaseOffset + modeOffset);
            Reg::CompareScalar<RangeType, CMPMODE::LT>(rightMask, rightPadIdxReg, lastDimIdx + modeOffset, maskIdx);
            Reg::Copy<RangeType, Reg::MaskMergeMode::MERGING>(lineRange, rightPadIdxReg, rightMask);

            // -2轴leftpad行数, 索引都一样
            __local_mem__ RangeType* idxAddrTmp = idxAddr;
            for (uint16_t i = 0; i < lastDimsLeft; i++) {
                RangeType loopStride = (RangeType)lastInDimSize *
                                       (lastDimsLeft - i - 1); // copyout需要带边界，故这里不区分mode
                Reg::Adds(lineRangeNew, lineRange, loopStride, maskIdx);
                Reg::DataCopyUnAlign(idxAddrTmp, lineRangeNew, uReg, lastOutDimSize);
            }
            Reg::DataCopyUnAlignPost(idxAddrTmp, uReg, 0);

            // -2轴有效输入的索引
            __local_mem__ RangeType* idxAddrTmp2 = idxAddr2;
            for (uint16_t i = 0; i < lastDimsLeft; i++) {
                RangeType loopStride = (RangeType)lastInDimSize * i;
                Reg::Adds(lineRangeNew, lineRange, loopStride, maskIdx);
                Reg::DataCopyUnAlign(idxAddrTmp2, lineRangeNew, uRegIn, lastOutDimSize);
            }
            Reg::DataCopyUnAlignPost(idxAddrTmp2, uRegIn, 0);
        }
    }

    __aicore__ inline void VlInCopyProc(uint16_t inLoops, uint16_t lastInLoops, RangeType idxOffset, uint32_t maskValue,
                                        uint32_t lastInMaskValue, __local_mem__ T* curInAddr,
                                        __local_mem__ T* curOutAddr, Reg::RegTensor<RangeType>& regIdx,
                                        uint32_t idxPadOffset)
    {
        Reg::RegTensor<T> regData;
        Reg::RegTensor<T> regDataT;
        Reg::RegTensor<RangeType> regIdxBk;
        Reg::RegTensor<RangeType> regNewIdx;
        Reg::UnalignReg uReg;
        uint32_t validMask = maskValue;
        if constexpr (sizeof(T) == 8) {
            validMask = maskValue * 2;
        }
        Reg::MaskReg maskIdx = Reg::UpdateMask<RangeType>(validMask);

        __local_mem__ T* outAddrTmp = curOutAddr;
        Reg::Adds(regIdxBk, regIdx, idxPadOffset, maskIdx);
        for (uint16_t cpIdx = 0; cpIdx < inLoops; cpIdx++) {
            Reg::Adds(regNewIdx, regIdxBk, cpIdx * idxOffset, maskIdx);
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curInAddr, (Reg::RegTensor<IdxType>&)regNewIdx,
                                maskIdx);
            if constexpr (sizeof(T) != 1) {
                Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, maskValue);
            } else {
                Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, maskValue);
            }
        }
        Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
        for (uint16_t cpTailIdx = 0; cpTailIdx < lastInLoops; cpTailIdx++) {
            outAddrTmp = curOutAddr + inLoops * maskValue;
            Reg::Adds(regNewIdx, regIdxBk, inLoops * idxOffset, maskIdx);
            Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curInAddr, (Reg::RegTensor<IdxType>&)regNewIdx,
                                maskIdx);
            if constexpr (sizeof(T) != 1) {
                Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, lastInMaskValue);
            } else {
                Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, lastInMaskValue);
            }
            Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
        }
    }

    // gather一次，copy多次
    __aicore__ inline void VlPaddingCopyProc(uint16_t gatherLoops, uint16_t padLoops, uint16_t lastPadLoops,
                                             uint32_t maskValue, uint32_t lastPadMaskValue,
                                             __local_mem__ T* curPadInAddr, __local_mem__ T* curPadOutAddr,
                                             Reg::RegTensor<RangeType>& regIdxPad, uint32_t idxPadOffset,
                                             RangeType idxOffset, uint16_t lastPadExcessIdx = 0)
    {
        Reg::RegTensor<T> regData;
        Reg::RegTensor<T> regDataT;
        Reg::RegTensor<RangeType> regIdxBk;
        Reg::RegTensor<RangeType> regNewIdx;
        Reg::RegTensor<RangeType> zeroIdxReg;
        Reg::Duplicate(zeroIdxReg, (RangeType)0);
        Reg::MaskReg zeroMask;
        Reg::UnalignReg uReg;
        uint32_t validMask = maskValue;
        uint32_t validLastMask = lastPadMaskValue;
        if constexpr (sizeof(T) == 8) {
            validMask = maskValue * 2;
            validLastMask = lastPadMaskValue * 2;
        }
        Reg::MaskReg maskIdx = Reg::UpdateMask<RangeType>(validMask);
        Reg::MaskReg maskLastIdx = Reg::UpdateMask<RangeType>(validLastMask);

        for (uint16_t gIdx = 0; gIdx < gatherLoops; gIdx++) {
            Reg::Adds(regIdxBk, regIdxPad, idxPadOffset, maskIdx);
            // 完整vl循环
            __local_mem__ T* outAddrTmp = curPadOutAddr + lastPadLoops * lastPadMaskValue;
            for (uint16_t pIdx = 0; pIdx < padLoops; pIdx++) {
                Reg::Adds(regNewIdx, regIdxBk, (padLoops - pIdx - 1) * idxOffset, maskIdx);
                // gather一次，copy多次
                Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curPadInAddr,
                                    (Reg::RegTensor<IdxType>&)regNewIdx, maskIdx);
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, maskValue);
                } else {
                    Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                    Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, maskValue);
                }
            }
            Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
            // vl 剩余的循环
            for (uint16_t pTaiIdx = 0; pTaiIdx < lastPadLoops; pTaiIdx++) {
                outAddrTmp = curPadOutAddr;
                Reg::Adds(regNewIdx, regIdxBk, padLoops * idxOffset - lastPadExcessIdx, maskLastIdx);
                Reg::DataCopyGather((Reg::RegTensor<CastType>&)regData, curPadInAddr,
                                    (Reg::RegTensor<IdxType>&)regNewIdx, maskLastIdx);
                if constexpr (sizeof(T) != 1) {
                    Reg::DataCopyUnAlign(outAddrTmp, regData, uReg, lastPadMaskValue);
                } else {
                    Reg::Pack(regDataT, (Reg::RegTensor<CastType>&)regData);
                    Reg::DataCopyUnAlign(outAddrTmp, regDataT, uReg, lastPadMaskValue);
                }
                Reg::DataCopyUnAlignPost(outAddrTmp, uReg, 0);
            }
        }
    }

    /*
    ub内二维，补正向的pad:
    1、只需要补尾轴的左右pad
    2、如果尾轴没有pad，直接copy到outTensor
    */
    __aicore__ inline void GatherProcessUb2DFw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        __local_mem__ RangeType* idxAddrBw = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddrFw = idxAddrBw + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr();

        uint16_t vlSplitLoopIn = vlSplitIn_;
        RangeType idxOffset = tdPtr_->inStride[dimNum_ - CONST2] * vlSplitLoopIn;
        uint32_t maskValue = tdPtr_->outStride[dimNum_ - CONST2] * vlSplitLoopIn;

        uint16_t copyInPadLoops = static_cast<uint16_t>(ubAxisInCopyNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(ubAxisInCopyNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * tdPtr_->outStride[dimNum_ - CONST2];

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdx, idxAddrFw);
            VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr, outAddr,
                         regIdx, 0);
        }
    }

    /*
    ub内二维，补反向的pad:
    0、计算当前拷贝的输入，是否在-2轴的上、下pad范围内，如果都不在，就直接返回
    1、补尾轴的左右pad
    2、-2轴倒序
    */
    __aicore__ inline void GatherProcessUb2DBw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        __local_mem__ RangeType* idxAddrBw = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddrFw = idxAddrBw + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + BLOCK_NUM * CONST2;

        uint16_t vlSplitLoopIn = vlSplitIn_;
        RangeType idxOffset = tdPtr_->inStride[dimNum_ - CONST2] * vlSplitLoopIn;
        uint32_t maskValue = tdPtr_->outStride[dimNum_ - CONST2] * vlSplitLoopIn;

        uint16_t copyInPadLoops = static_cast<uint16_t>(ubAxisInCopyNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(ubAxisInCopyNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * tdPtr_->outStride[dimNum_ - CONST2];
        uint16_t lastPadExcessIdx = lastCopyInPadNum == 0 ?
                                        0 :
                                        (maskValue - lastCopyInMaskValue) * tdPtr_->inStride[dimNum_ - CONST2] /
                                            tdPtr_->outStride[dimNum_ - CONST2];

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdxPad;
            Reg::DataCopy(regIdxPad, idxAddrBw);

            VlPaddingCopyProc(1, copyInPadLoops, lastCopyInPadLoops, maskValue, lastCopyInMaskValue, inAddr, outAddr,
                              regIdxPad, 0, idxOffset, lastPadExcessIdx);
        }
    }

    template <bool IS_FW>
    __aicore__ inline void GatherProcessUb3D(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                             LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        bool isFw = false;
        if constexpr (IS_FW) {
            isFw = true;
        }
        __local_mem__ RangeType* idxAddrBw = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddrFw = idxAddrBw + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + (isFw ? 0 : BLOCK_NUM * CONST2);

        uint16_t vlSplitLoopIn = vlSplitIn_;
        uint32_t strideInVl = tdPtr_->inStride[dimNum_ - CONST2];
        uint32_t strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST3];
        uint32_t strideOutVl = tdPtr_->outStride[dimNum_ - CONST2];
        uint32_t strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST3];

        // 该次Ub内C轴左pad, VL切3维时退化为1
        uint16_t ubAxisInCopyLoops = ubAxisInCopyNum;

        // ub切-3，这个值不会很大，uint32_t足够
        uint32_t vlLeftPadNum = tdPtr_->leftPad[dimNum_ - CONST2];
        uint32_t vlInNum = tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t vlRightPadNum = tdPtr_->outShape[dimNum_ - CONST2] - tdPtr_->leftPad[dimNum_ - CONST2] -
                                 tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t rightPadInVlOffset = vlInNum == 0 ? 0 : (vlInNum - vlRightPadNum - 1) * strideInVl;

        if (lastThirdDimInVL_) {
            strideInVl = tdPtr_->inStride[dimNum_ - CONST3];
            strideInVlO1 = 1;
            strideOutVl = tdPtr_->outStride[dimNum_ - CONST3];
            strideOutVlO1 = 1;
            ubAxisInCopyLoops = 1;
            vlLeftPadNum = isFw ? 0 : ubAxisInCopyNum;
            vlInNum = isFw ? ubAxisInCopyNum : 0;
            vlRightPadNum = 0;
        }

        RangeType idxOffset = strideInVl * vlSplitLoopIn;
        uint32_t maskValue = strideOutVl * vlSplitLoopIn;

        uint16_t leftGatherLoops = vlLeftPadNum == 0 ? 0 : 1;
        uint16_t leftPadLoops = static_cast<uint16_t>(vlLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(vlLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * strideOutVl;
        uint16_t lastLeftPadExcessIdx = lastLeftPadNum == 0 ?
                                            0 :
                                            (maskValue - lastLeftPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST2] /
                                                tdPtr_->outStride[dimNum_ - CONST2];

        uint16_t copyInPadLoops = static_cast<uint16_t>(vlInNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(vlInNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * strideOutVl;

        uint16_t rightGatherLoops = vlRightPadNum == 0 ? 0 : 1;
        uint16_t rightPadLoops = static_cast<uint16_t>(vlRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum = static_cast<uint16_t>(vlRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum * strideOutVl;
        uint16_t lastRightPadExcessIdx = lastrightPadNum == 0 ?
                                             0 :
                                             (maskValue - lastrightPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST2] /
                                                 tdPtr_->outStride[dimNum_ - CONST2];

        uint32_t modeOffset = 0; // fw需要全量输入的反向，故idx带了边界，按mode增加处理
        if constexpr (IS_REFLECT) {
            modeOffset = strideInVl;
        } else {
            rightPadInVlOffset = vlInNum == 0 ? 0 : (vlInNum - vlRightPadNum) * strideInVl;
        }

        if (lastThirdDimInVL_) {
            modeOffset = 0;
            lastLeftPadExcessIdx = lastLeftPadNum == 0 ?
                                       0 :
                                       (maskValue - lastLeftPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST3] /
                                           tdPtr_->outStride[dimNum_ - CONST3];
            lastRightPadExcessIdx = lastrightPadNum == 0 ?
                                        0 :
                                        (maskValue - lastrightPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST3] /
                                            tdPtr_->outStride[dimNum_ - CONST3];
        }

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdxPad;
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdxPad, idxAddrBw);
            Reg::DataCopy(regIdx, idxAddrFw);
            Reg::RegTensor<RangeType> regIdxPadFixed;

            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::Adds(regIdxPadFixed, regIdxPad, modeOffset, maskIdx);

            // 该次Ub内C轴上的copyIn, VL切3维时退化为0
            for (uint16_t uiIdx = 0; uiIdx < ubAxisInCopyLoops; uiIdx++) {
                uint16_t fixedIdx = 0;
                if constexpr (IS_FW) {
                    fixedIdx = uiIdx;
                } else {
                    fixedIdx = (ubAxisInCopyLoops - 1 - uiIdx);
                }
                // H轴左pad
                uint32_t idxPadOffset = uiIdx * strideInVlO1;
                __local_mem__ T* curInOutAddr = outAddr + fixedIdx * strideOutVlO1;
                VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                  inAddr, curInOutAddr, regIdxPadFixed, idxPadOffset, idxOffset, lastLeftPadExcessIdx);

                // copyInPad -- gather多次 copy多次
                curInOutAddr = outAddr + fixedIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl;
                VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                             curInOutAddr, regIdx, idxPadOffset);

                // right pad -- gather一次，copy多次
                curInOutAddr = outAddr + fixedIdx * strideOutVlO1 + vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                idxPadOffset = uiIdx * strideInVlO1 + rightPadInVlOffset;
                VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue, lastrightPadMaskValue,
                                  inAddr, curInOutAddr, regIdxPad, idxPadOffset, idxOffset, lastRightPadExcessIdx);
            }
        }
    }

    /*
    ub内三维，ub切-3轴，vl切-2轴或-3轴，补正向的pad:
    1、补尾轴的pad
    2、补-2轴的pad
    */
    __aicore__ inline void GatherProcessUb3DFw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        GatherProcessUb3D<true>(idxTensor, inTensor, outTensor, ubAxisInCopyNum);
    }

    /*
    ub内三维，补反向的pad:
    0、计算当前拷贝的输入，是否在-3轴的上、下pad范围内，如果都不在，就直接返回
    1、补尾轴的左右pad
    2、补-2轴的pad
    3、-3轴倒序
    */
    __aicore__ inline void GatherProcessUb3DBw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        GatherProcessUb3D<false>(idxTensor, inTensor, outTensor, ubAxisInCopyNum);
    }

    template <bool IS_FW>
    __aicore__ inline void GatherProcessUb4D(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                             LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        bool isFw = false;
        if constexpr (IS_FW) {
            isFw = true;
        }
        __local_mem__ RangeType* idxPadAddr = (__local_mem__ RangeType*)idxTensor.GetPhyAddr();
        __local_mem__ RangeType* idxAddr = idxPadAddr + VL_RANGE_CNT;
        __local_mem__ T* inAddr = (__local_mem__ T*)inTensor.GetPhyAddr();
        __local_mem__ T* outAddr = (__local_mem__ T*)outTensor.GetPhyAddr() + (isFw ? 0 : BLOCK_NUM * CONST2);

        uint16_t vlSplitLoopIn = vlSplitIn_;
        uint16_t ubAxisLeftPadLoops = 0;              // ubAxisLeftPadNum; 当前不支持N轴的pad, 只会为0
        uint16_t ubAxisInCopyLoops = ubAxisInCopyNum; // ubAxisInCopyNum;
        uint16_t ubAxisRightPadLoops = 0;             // ubAxisRightPadNum; 当前不支持N轴的pad, 只会为0

        uint32_t strideInVl = tdPtr_->inStride[dimNum_ - CONST2];
        uint32_t strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST3];
        uint32_t strideOutVl = tdPtr_->outStride[dimNum_ - CONST2];
        uint32_t strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST3];

        // ub切-4，这个值不会很大，uint32_t足够
        uint16_t vlO1LeftPadNum = tdPtr_->leftPad[dimNum_ - CONST3];
        uint16_t vlO1InNum = tdPtr_->inShape[dimNum_ - CONST3];
        uint16_t vlO1RightPadNum = tdPtr_->outShape[dimNum_ - CONST3] - tdPtr_->leftPad[dimNum_ - CONST3] -
                                   tdPtr_->inShape[dimNum_ - CONST3];

        uint32_t vlLeftPadNum = tdPtr_->leftPad[dimNum_ - CONST2];
        uint32_t vlInNum = tdPtr_->inShape[dimNum_ - CONST2];
        uint32_t vlRightPadNum = tdPtr_->outShape[dimNum_ - CONST2] - tdPtr_->leftPad[dimNum_ - CONST2] -
                                 tdPtr_->inShape[dimNum_ - CONST2];

        if (lastThirdDimInVL_) {
            strideInVl = tdPtr_->inStride[dimNum_ - CONST3];
            strideInVlO1 = tdPtr_->inStride[dimNum_ - CONST4];
            strideOutVl = tdPtr_->outStride[dimNum_ - CONST3];
            strideOutVlO1 = tdPtr_->outStride[dimNum_ - CONST4];

            vlLeftPadNum = tdPtr_->leftPad[dimNum_ - CONST3];
            vlInNum = tdPtr_->inShape[dimNum_ - CONST3];
            vlRightPadNum = tdPtr_->outShape[dimNum_ - CONST3] - tdPtr_->leftPad[dimNum_ - CONST3] -
                            tdPtr_->inShape[dimNum_ - CONST3];

            vlO1LeftPadNum = 0;
            vlO1InNum = 1;
            vlO1RightPadNum = 0;
        }

        RangeType idxOffset = strideInVl * vlSplitLoopIn;
        uint32_t maskValue = strideOutVl * vlSplitLoopIn;

        uint32_t strideInN = tdPtr_->inStride[dimNum_ - CONST4];
        uint32_t strideOutN = tdPtr_->outStride[dimNum_ - CONST4];

        uint16_t leftGatherLoops = vlLeftPadNum == 0 ? 0 : 1;
        uint16_t leftPadLoops = static_cast<uint16_t>(vlLeftPadNum / vlSplitLoopIn);
        uint16_t lastLeftPadNum = static_cast<uint16_t>(vlLeftPadNum - leftPadLoops * vlSplitLoopIn);
        uint16_t lastLeftPadLoops = lastLeftPadNum == 0 ? 0 : 1;
        uint32_t lastLeftPadMaskValue = lastLeftPadNum * strideOutVl;
        uint16_t lastLeftPadExcessIdx = lastLeftPadNum == 0 ?
                                            0 :
                                            (maskValue - lastLeftPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST2] /
                                                tdPtr_->outStride[dimNum_ - CONST2];

        uint16_t copyInPadLoops = static_cast<uint16_t>(vlInNum / vlSplitLoopIn);
        uint16_t lastCopyInPadNum = static_cast<uint16_t>(vlInNum - copyInPadLoops * vlSplitLoopIn);
        uint16_t lastCopyInPadLoops = lastCopyInPadNum == 0 ? 0 : 1;
        uint32_t lastCopyInMaskValue = lastCopyInPadNum * strideOutVl;

        uint16_t rightGatherLoops = vlRightPadNum == 0 ? 0 : 1;
        uint16_t rightPadLoops = static_cast<uint16_t>(vlRightPadNum / vlSplitLoopIn);
        uint16_t lastrightPadNum = static_cast<uint16_t>(vlRightPadNum - rightPadLoops * vlSplitLoopIn);
        uint16_t lastrightPadLoops = lastrightPadNum == 0 ? 0 : 1;
        uint32_t lastrightPadMaskValue = lastrightPadNum * strideOutVl;
        uint16_t lastRightPadExcessIdx = lastrightPadNum == 0 ?
                                             0 :
                                             (maskValue - lastrightPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST2] /
                                                 tdPtr_->outStride[dimNum_ - CONST2];

        uint32_t modeHOffset = 0; // 适配fw index包含边界，实际使用时在reflect下需要做处理
        uint32_t modeCOffset = 0;
        uint32_t rightPadInVlOffset = (vlInNum - vlRightPadNum - 1) * strideInVl;
        uint32_t rightPadInVlO1Offset = (vlO1InNum - vlO1RightPadNum - 1) * strideInVlO1;
        if constexpr (IS_REFLECT) {
            modeHOffset = strideInVl;
            modeCOffset = strideInVlO1;
        } else {
            rightPadInVlOffset = (vlInNum - vlRightPadNum) * strideInVl;
            rightPadInVlO1Offset = (vlO1InNum - vlO1RightPadNum) * strideInVlO1;
        }

        uint32_t modeCOffset2 = 0;
        if (lastThirdDimInVL_) {
            // 此场景，modeHOffset为0，modeCOffset不参与计算，modeCOffset2为C轴左Pad偏移边界1
            modeHOffset = 0;
            if constexpr (IS_REFLECT) {
                modeCOffset2 = strideInVl;
            }
            lastLeftPadExcessIdx = lastLeftPadNum == 0 ?
                                       0 :
                                       (maskValue - lastLeftPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST3] /
                                           tdPtr_->outStride[dimNum_ - CONST3];
            lastRightPadExcessIdx = lastrightPadNum == 0 ?
                                        0 :
                                        (maskValue - lastrightPadMaskValue) * tdPtr_->inStride[dimNum_ - CONST3] /
                                            tdPtr_->outStride[dimNum_ - CONST3];
        }

        __VEC_SCOPE__
        {
            Reg::RegTensor<RangeType> regIdxPad;
            Reg::RegTensor<RangeType> regIdx;
            Reg::DataCopy(regIdxPad, idxPadAddr);
            Reg::DataCopy(regIdx, idxAddr);
            Reg::RegTensor<RangeType> regIdxPadFixed;

            Reg::MaskReg maskIdx = Reg::CreateMask<RangeType, Reg::MaskPattern::ALL>();
            Reg::Adds(regIdxPadFixed, regIdxPad, modeHOffset, maskIdx);

            for (uint16_t nIdx = 0; nIdx < ubAxisInCopyLoops; nIdx++) {
                uint16_t fixedIdx = 0;
                if constexpr (IS_FW) {
                    fixedIdx = nIdx;
                } else {
                    fixedIdx = (ubAxisInCopyLoops - 1 - nIdx);
                }
                // C轴左pad, VL切3维时退化为0
                uint32_t curInOffset = nIdx * strideInN + modeCOffset;
                uint32_t curOutOffset = fixedIdx * strideOutN;
                for (uint16_t i = 0; i < vlO1LeftPadNum; i++) {
                    // H轴左pad
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + (vlO1LeftPadNum - 1 - i) * strideOutVlO1;
                    VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                      inAddr, curPadOutAddr, regIdxPadFixed, idxPadOffset, idxOffset,
                                      lastLeftPadExcessIdx);

                    // H轴输入个数
                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + (vlO1LeftPadNum - 1 - i) * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    // H轴右pad
                    idxPadOffset = curInOffset + i * strideInVlO1 + rightPadInVlOffset;
                    curPadOutAddr = outAddr + curOutOffset + (vlO1LeftPadNum - 1 - i) * strideOutVlO1 +
                                    vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                    VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue,
                                      lastrightPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset, idxOffset,
                                      lastRightPadExcessIdx);
                }
                // C轴上的输入个数，VL切3维时退化为1
                curInOffset = nIdx * strideInN;
                curOutOffset = fixedIdx * strideOutN + vlO1LeftPadNum * strideOutVlO1;
                for (uint16_t i = 0; i < vlO1InNum; i++) {
                    // H轴左pad, VL切3维时退化为C轴左pad
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1 + modeCOffset2;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1;
                    VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                      inAddr, curPadOutAddr, regIdxPadFixed, idxPadOffset, idxOffset,
                                      lastLeftPadExcessIdx);

                    // H轴输入有效个数, VL切3维时退化为C轴输入个数
                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + i * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    idxPadOffset = curInOffset + i * strideInVlO1;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    // H轴右pad, VL切3维时退化为C轴右pad
                    idxPadOffset = curInOffset + i * strideInVlO1 + rightPadInVlOffset;
                    curPadOutAddr = outAddr + curOutOffset + i * strideOutVlO1 + vlLeftPadNum * strideOutVl +
                                    vlInNum * strideOutVl;
                    VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue,
                                      lastrightPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset, idxOffset,
                                      lastRightPadExcessIdx);
                }
                // C轴右pad，VL切3维时退化为0
                curInOffset = nIdx * strideInN + rightPadInVlO1Offset;
                curOutOffset = fixedIdx * strideOutN + vlO1LeftPadNum * strideOutVlO1 + vlO1InNum * strideOutVlO1;
                for (uint16_t i = 0; i < vlO1RightPadNum; i++) {
                    uint32_t idxPadOffset = curInOffset + i * strideInVlO1;
                    __local_mem__ T* curPadOutAddr = outAddr + curOutOffset + (vlO1RightPadNum - 1 - i) * strideOutVlO1;
                    VlPaddingCopyProc(leftGatherLoops, leftPadLoops, lastLeftPadLoops, maskValue, lastLeftPadMaskValue,
                                      inAddr, curPadOutAddr, regIdxPadFixed, idxPadOffset, idxOffset,
                                      lastLeftPadExcessIdx);

                    __local_mem__ T* curOutAddr = outAddr + curOutOffset + (vlO1RightPadNum - 1 - i) * strideOutVlO1 +
                                                  vlLeftPadNum * strideOutVl;
                    VlInCopyProc(copyInPadLoops, lastCopyInPadLoops, idxOffset, maskValue, lastCopyInMaskValue, inAddr,
                                 curOutAddr, regIdx, idxPadOffset);

                    idxPadOffset = curInOffset + i * strideInVlO1 + rightPadInVlOffset;
                    curPadOutAddr = outAddr + curOutOffset + (vlO1RightPadNum - 1 - i) * strideOutVlO1 +
                                    vlLeftPadNum * strideOutVl + vlInNum * strideOutVl;
                    VlPaddingCopyProc(rightGatherLoops, rightPadLoops, lastrightPadLoops, maskValue,
                                      lastrightPadMaskValue, inAddr, curPadOutAddr, regIdxPad, idxPadOffset, idxOffset,
                                      lastRightPadExcessIdx);
                }
            }
        }
    }

    /*
    ub内四维，NCHW: ub切N轴，vl切H轴或C轴。当前仅支持CHW维的pad，补正向的pad:
    1、补尾轴的pad
    2、补-2轴的pad
    3、补-3轴的pad
    */
    __aicore__ inline void GatherProcessUb4DFw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        GatherProcessUb4D<true>(idxTensor, inTensor, outTensor, ubAxisInCopyNum);
    }

    __aicore__ inline void GatherProcessUb4DBw(const LocalTensor<RangeType>& idxTensor, LocalTensor<T>& inTensor,
                                               LocalTensor<T>& outTensor, uint16_t ubAxisInCopyNum)
    {
        GatherProcessUb4D<false>(idxTensor, inTensor, outTensor, ubAxisInCopyNum);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetWaitEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }
};
} // namespace PadV3

#endif
