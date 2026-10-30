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
 * \file batch_to_space_largec.h
 * \brief batch_to_space_largec
 */

#ifndef BATCH_TO_SPACE_LARGEC_H
#define BATCH_TO_SPACE_LARGEC_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "batch_to_space_nd_tiling_data.h"

namespace B2SND {
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
class BatchToSpaceLargeC {
private:
    TPipe* pipe_ = nullptr;
    const B2SNDLargeCTilingData* tdPtr_ = nullptr;

    // buffer
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> inQueue_;

    uint32_t blockIdx_{0};
    uint32_t rank_{0};
    uint32_t blockShapeSize_{0};
    uint64_t blockShapeProduct_{1};
    uint8_t ubAxis_{0};
    uint32_t ubFactor_{0};
    int32_t outputTileSize_{0};
    uint64_t originC_, alignC_;

    // tileQueryNums矩阵维度 totalCount_/N',切H时用该矩阵来索引。
    uint32_t tileQueryNumsDim_{0};
    // 整块切分内部切分矩阵维度
    uint32_t innerTileNumsDim_{0};
    uint32_t headTilesDim_{0};
    uint32_t tailTilesDim_{0};
    uint32_t ubRealFactorTotal_{0};

    uint32_t alignedCLength_{0};

    uint64_t totalCount_{0};
    uint64_t perCoreCount_{0};
    uint32_t axisPreProduct_{1};
    uint64_t inStride_[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t outStride_[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t instrideBS[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t cropHtop_{0};
    uint64_t cropHbottom_{0};
    uint64_t BSH_{0};
    uint64_t outH_{0};
    uint64_t headLenH_{0};
    uint64_t middleLenH_{0};
    uint64_t tailLenH_{0};

    uint64_t cropLeft_{0};
    uint64_t cropRight_{0};
    uint64_t BSW_{0};
    uint64_t outW_{0};
    uint64_t inW_{0};
    uint64_t inC_{0};
    uint64_t leftCopyLen_{0};
    uint64_t middleCopyLen_{0};
    uint64_t rightCopyLen_{0};

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t UB_BLOCK = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    // Axis offset constants
    constexpr static uint32_t AXIS_OFFSET_W = 2;       // W轴相对rank末尾的偏移量
    constexpr static uint32_t AXIS_OFFSET_H = 3;       // H轴相对rank末尾的偏移量
    constexpr static uint32_t BLOCK_AXIS_OFFSET_H = 2; // H轴在blockShape数组中的偏移量
    // Threshold/rank constants
    constexpr static uint32_t MIN_SPATIAL_DIMS = 2;    // 处理H轴所需的最少空间维度数
    constexpr static uint32_t MIN_RANK_FOR_H_AXIS = 4; // 处理H轴所需的最小tensor rank
    constexpr static uint32_t RANK_3D = 3;             // 3维tensor的rank值
    constexpr static uint32_t RANK_5D = 5;             // 5维tensor的rank值

public:
    __aicore__ inline BatchToSpaceLargeC(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const B2SNDLargeCTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tdPtr_ = tilingData;
        rank_ = tdPtr_->input.rank; // N` + M + C >2：等于2时remain_shape为0，进gather模板；<6：block_shape大于3时走SIMT
        blockShapeSize_ = rank_ - 2; // M
        ubAxis_ = tdPtr_->ubAxis;
        ubFactor_ = tdPtr_->ubFactor;

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);
        outputTileSize_ = tdPtr_->outputBufferSize / BUFFER_NUM;
        pipe_->InitBuffer(inQueue_, tdPtr_->outputBufferSize);

        alignedCLength_ = CeilAlign(static_cast<uint32_t>(tdPtr_->input.outShape[rank_ - 1]), BLK_ELEMS);
        totalCount_ = tdPtr_->totalCount;
        perCoreCount_ = tdPtr_->perCoreCount;

        for (int i = ubAxis_ - 1; i >= 0; i--) {
            axisPreProduct_ *= tdPtr_->input.outShape[i];
        }
        tileQueryNumsDim_ = totalCount_ / axisPreProduct_;
        if (blockShapeSize_ >= MIN_SPATIAL_DIMS) {
            // H 轴
            cropHtop_ = tdPtr_->input.crops[blockShapeSize_ - BLOCK_AXIS_OFFSET_H][0];
            cropHbottom_ = tdPtr_->input.crops[blockShapeSize_ - BLOCK_AXIS_OFFSET_H][1];
            BSH_ = tdPtr_->input.blockShape[blockShapeSize_ - BLOCK_AXIS_OFFSET_H];
            outH_ = tdPtr_->input.outShape[rank_ - AXIS_OFFSET_H];

            CalcBoundaryBlock(cropHtop_, cropHbottom_, BSH_, outH_, headLenH_, tailLenH_, middleLenH_);

            headTilesDim_ = Ops::Base::CeilDiv(headLenH_, static_cast<uint64_t>(ubFactor_));
            tailTilesDim_ = Ops::Base::CeilDiv(tailLenH_, static_cast<uint64_t>(ubFactor_));

            if (middleLenH_ > 0) {
                if (ubFactor_ < BSH_) {
                    innerTileNumsDim_ = Ops::Base::CeilDiv(BSH_, static_cast<uint64_t>(ubFactor_));
                    ubRealFactorTotal_ = BSH_;
                } else {
                    innerTileNumsDim_ = tileQueryNumsDim_ - headTilesDim_ - tailTilesDim_;
                    ubRealFactorTotal_ = outH_ - headLenH_ % ubFactor_ - tailLenH_ % ubFactor_;
                }
            } else {
                innerTileNumsDim_ = 0;
                ubRealFactorTotal_ = 0;
            }

            // ===================== W 轴 初始化 =====================
            cropLeft_ = tdPtr_->input.crops[blockShapeSize_ - 1][0];
            cropRight_ = tdPtr_->input.crops[blockShapeSize_ - 1][1];
            BSW_ = tdPtr_->input.blockShape[blockShapeSize_ - 1];
            outW_ = tdPtr_->input.outShape[rank_ - AXIS_OFFSET_W];
            inW_ = tdPtr_->input.inShape[rank_ - AXIS_OFFSET_W];
            inC_ = tdPtr_->input.inShape[rank_ - 1];

            CalcBoundaryBlock(cropLeft_, cropRight_, BSW_, outW_, leftCopyLen_, rightCopyLen_, middleCopyLen_);
        }
    }

    __aicore__ inline void Process()
    {
        // 空tensor直接返回
        for (uint8_t i = 0; i < rank_; i++) {
            if (tdPtr_->input.outShape[i] == 0) {
                return;
            }
        }

        uint32_t startIdx = blockIdx_ * perCoreCount_;
        if (startIdx >= totalCount_) {
            return;
        }

        uint64_t inShapeSize = 1UL;
        uint64_t outShapeSize = 1UL;
        for (int64_t i = rank_ - 1; i >= 0; --i) {
            // compute in stride
            inStride_[i] = inShapeSize;
            inShapeSize *= tdPtr_->input.inShape[i];
            // compute out stride
            outStride_[i] = outShapeSize;
            outShapeSize *= tdPtr_->input.outShape[i];
        }
        instrideBS[0] = inShapeSize;
        for (uint8_t i = 0; i < blockShapeSize_; i++) {
            blockShapeProduct_ *= tdPtr_->input.blockShape[i];
            instrideBS[i + 1] = inShapeSize / blockShapeProduct_;
        }

        originC_ = tdPtr_->input.inShape[rank_ - 1];
        alignC_ = Ops::Base::CeilAlign(originC_, static_cast<uint64_t>(BLK_ELEMS));
        LocalTensor<T> input = inQueue_.Get<T>();
        uint32_t endIdx = (blockIdx_ + 1L) * perCoreCount_;
        endIdx = endIdx < totalCount_ ? endIdx : totalCount_;

        if (ubAxis_ == rank_ - AXIS_OFFSET_H && (rank_ == MIN_RANK_FOR_H_AXIS || rank_ == RANK_5D)) {
            int sum = getTileNumPrefixSum(startIdx);

            uint64_t curOutIndex[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t outIndex[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            uint64_t inIndex[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

            convert(outIndex, sum);

            // Update outIndex for next iteration
            for (uint8_t i = 0; i < rank_; i++) {
                curOutIndex[i] = outIndex[i];
            }
            for (uint32_t idx = startIdx; idx < endIdx; idx++) {
                uint32_t ubAxisInCopyNum = 0;
                LocalTensor<T> srcLocal = input[((idx - startIdx) & (BUFFER_NUM - 1)) * outputTileSize_ / sizeof(T)];
                // CopyIn
                CopyIn(srcLocal, curOutIndex, inIndex, ubAxisInCopyNum, idx, (idx - startIdx));
                SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                // CopyOut
                CopyOut(srcLocal, outIndex, ubAxisInCopyNum);
                SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);

                // Update outIndex for next iteration
                for (uint8_t i = 0; i < rank_; i++) {
                    outIndex[i] = curOutIndex[i];
                }
            }
        } else {
            for (uint32_t idx = startIdx; idx < endIdx; idx++) {
                uint64_t outIndex[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                uint64_t inIndex[MAX_INPUT_RANK] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
                uint32_t ubAxisInCopyNum = 0;

                LocalTensor<T> srcLocal = input[((idx - startIdx) & (BUFFER_NUM - 1)) * outputTileSize_ / sizeof(T)];
                // CopyIn
                CopyIn(srcLocal, outIndex, inIndex, ubAxisInCopyNum, idx, (idx - startIdx));
                SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
                // CopyOut
                CopyOut(srcLocal, outIndex, ubAxisInCopyNum);
                SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        }
    }

private:
    __aicore__ inline void CalculateOutIndex(uint32_t curIdx, uint64_t* outIndex)
    {
        for (int8_t i = ubAxis_; i >= 0; i--) { // Changed from uint8_t to int8_t to handle potential negative
            uint64_t factor = tdPtr_->input.outShape[i];
            if (i == ubAxis_) {
                factor = Ops::Base::CeilDiv(factor, static_cast<uint64_t>(ubFactor_));
            }
            if (factor != 0) {
                outIndex[i] = (i == ubAxis_ ? curIdx % factor * ubFactor_ : curIdx % factor);
            }
            if (factor != 0) {
                curIdx = curIdx / factor;
            }
        }
    }

    __aicore__ inline void Conver2InIndex(uint64_t* outIndex, uint64_t* inIndex)
    {
        // transalate outIndex to inIndex
        // compute inIndex[1..M] and blockOffset
        uint64_t blockOffset = 0;
        for (uint8_t i = 0; i < blockShapeSize_; i++) {
            uint64_t u = outIndex[i + 1] + tdPtr_->input.crops[i][0];
            uint64_t b = tdPtr_->input.blockShape[i];
            uint64_t t = u % b;
            inIndex[i + 1] = u / b;
            blockOffset = blockOffset * b + t;
        }
        // N
        inIndex[0] = blockOffset * tdPtr_->input.outShape[0] + outIndex[0];
        // C
        inIndex[rank_ - 1] = outIndex[rank_ - 1];
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, uint64_t* outIndex, uint64_t* inIndex,
                                  uint32_t& ubAxisInCopyNum, uint32_t idx, uint32_t x)
    {
        if (ubAxis_ >= rank_) {
            return;
        }
        if (x >= 1) {
            SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        }

        if (ubAxis_ == rank_ - 1) {
            DoCopyInAxisC(src, outIndex, inIndex, ubAxisInCopyNum, idx);
        } else if (ubAxis_ == rank_ - AXIS_OFFSET_W) {
            DoCopyInAxisW(src, outIndex, inIndex, ubAxisInCopyNum, idx);
        } else if (ubAxis_ == rank_ - AXIS_OFFSET_H && rank_ == RANK_3D) {
            DoCopyInAxisN3(src, outIndex, inIndex, ubAxisInCopyNum, idx);
        } else if (ubAxis_ == rank_ - AXIS_OFFSET_H && rank_ >= MIN_RANK_FOR_H_AXIS) { // H轴
            DoCopyInAxisH(src, outIndex, inIndex, ubAxisInCopyNum, idx);
        }
    }

    __aicore__ inline void DoCopyInAxisC(const LocalTensor<T>& src, uint64_t* outIndex, uint64_t* inIndex,
                                         uint32_t& ubAxisInCopyNum, uint32_t idx)
    {
        CalculateOutIndex(idx, outIndex);
        Conver2InIndex(outIndex, inIndex);

        uint64_t ubTailFactor = tdPtr_->input.outShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        ubAxisInCopyNum = (outIndex[ubAxis_] + ubFactor_ > tdPtr_->input.outShape[ubAxis_]) ? ubTailFactor : ubFactor_;

        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < rank_; i++) {
            inAddr += inIndex[i] * inStride_[i];
        }
        uint32_t copyInNum = ubAxisInCopyNum * inStride_[ubAxis_]; // 此时stride为1
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = copyInNum * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(src, inputGm_[inAddr], copyInParams, padParams);
    }

    __aicore__ inline void DoCopyInAxisW(const LocalTensor<T>& src, uint64_t* outIndex, uint64_t* inIndex,
                                         uint32_t& ubAxisInCopyNum, uint32_t idx)
    {
        CalcFactorAxisW(outIndex, ubAxisInCopyNum, idx);
        Conver2InIndex(outIndex, inIndex);

        uint64_t inAddr = 0;
        uint32_t strideC = tdPtr_->input.outShape[0];
        for (uint8_t i = 0; i < rank_; i++) {
            inAddr += inIndex[i] * inStride_[i];
            if (i != 0) {
                strideC *= tdPtr_->input.inShape[i];
            }
        }

        uint64_t wBlockShape = tdPtr_->input.blockShape[blockShapeSize_ - 1];
        bool cond = ubAxisInCopyNum <= wBlockShape;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = cond ? ubAxisInCopyNum : wBlockShape;
        copyInParams.blockLen = originC_ * sizeof(T);
        copyInParams.srcStride = (strideC - originC_) * sizeof(T);
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = cond ? 1 : ubAxisInCopyNum / wBlockShape;
        loopParams.loop1SrcStride = originC_ * sizeof(T);
        loopParams.loop1DstStride = alignC_ * wBlockShape * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(src, inputGm_[inAddr], copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void GetAxisLoop(uint8_t ubAxis, uint32_t ubFactor, uint32_t& headLoop, uint32_t& headMode,
                                       uint32_t& centerLoop, uint32_t& centerMode, uint32_t& centerBSLoop,
                                       uint32_t& tailLoop, uint32_t& tailMode, uint32_t& totalLoop)
    {
        uint32_t tmp = tdPtr_->input.crops[ubAxis - 1][0] % tdPtr_->input.blockShape[ubAxis - 1];
        uint32_t headCountAxis = tmp == 0 ? 0 : tdPtr_->input.blockShape[ubAxis - 1] - tmp;
        tmp = tdPtr_->input.crops[ubAxis - 1][1] % tdPtr_->input.blockShape[ubAxis - 1];
        uint32_t tailCountAxis = tmp == 0 ? 0 : tdPtr_->input.blockShape[ubAxis - 1] - tmp;

        if (headCountAxis + tailCountAxis > tdPtr_->input.outShape[ubAxis]) {
            uint32_t centerCountAxis = tdPtr_->input.outShape[ubAxis];
            centerLoop = Ops::Base::CeilDiv(centerCountAxis, ubFactor);
            centerMode = centerCountAxis % ubFactor;
            totalLoop = centerLoop;
            return;
        }

        uint32_t centerCountAxis = tdPtr_->input.outShape[ubAxis] - headCountAxis - tailCountAxis;

        headLoop = Ops::Base::CeilDiv(headCountAxis, ubFactor);
        headMode = headCountAxis % ubFactor;
        centerLoop = Ops::Base::CeilDiv(centerCountAxis, ubFactor);
        centerMode = centerCountAxis % ubFactor;
        if (tdPtr_->input.blockShape[ubAxis - 1] > ubFactor) {
            centerBSLoop = centerCountAxis / tdPtr_->input.blockShape[ubAxis - 1];
            centerLoop = centerBSLoop *
                         Ops::Base::CeilDiv(static_cast<uint32_t>(tdPtr_->input.blockShape[ubAxis - 1]), ubFactor);
            centerMode = tdPtr_->input.blockShape[ubAxis - 1] % ubFactor;
        }
        tailLoop = Ops::Base::CeilDiv(tailCountAxis, ubFactor);
        tailMode = tailCountAxis % ubFactor;
        totalLoop = headLoop + centerLoop + tailLoop;
    }

    __aicore__ inline void CalcFactorAxisW(uint64_t* outIndex, uint32_t& ubAxisInCopyNum, uint32_t idx)
    {
        uint32_t wSize = tdPtr_->input.outShape[ubAxis_];
        uint32_t wFactor = ubFactor_;
        uint32_t headLoop{0}, headMode{0}, centerLoop{0}, centerMode{0}, centerBSLoop{1}, tailLoop{0}, tailMode{0},
            wLoop{0};
        GetAxisLoop(ubAxis_, wFactor, headLoop, headMode, centerLoop, centerMode, centerBSLoop, tailLoop, tailMode,
                    wLoop);

        uint32_t wIndex = idx % wLoop;
        GetCopyInNum(ubAxisInCopyNum, wIndex, headLoop, headMode, centerLoop, centerMode, tailLoop, tailMode,
                     centerBSLoop, wFactor);

        // calc outIndex
        uint32_t totalWIndex = idx / wLoop * wSize;
        uint32_t tmpCopyNum = 0;
        for (int i = 0; i < wIndex; i++) {
            GetCopyInNum(tmpCopyNum, i, headLoop, headMode, centerLoop, centerMode, tailLoop, tailMode, centerBSLoop,
                         wFactor);
            totalWIndex += tmpCopyNum;
        }

        uint64_t totalIndex = totalWIndex * tdPtr_->input.outShape[rank_ - 1];
        for (int8_t i = rank_ - 1; i >= 0; i--) {
            outIndex[i] = totalIndex % tdPtr_->input.outShape[i];
            totalIndex /= tdPtr_->input.outShape[i];
        }
    }

    __aicore__ inline void DoCopyInAxisN3(const LocalTensor<T>& src, uint64_t* outIndex, uint64_t* inIndex,
                                          uint32_t& ubAxisInCopyNum, uint32_t idx)
    {
        uint64_t ubTailFactor = tdPtr_->input.outShape[ubAxis_] % ubFactor_;
        ubTailFactor = (ubTailFactor == 0) ? ubFactor_ : ubTailFactor;
        ubAxisInCopyNum = (idx == tdPtr_->totalCount - 1) ? ubTailFactor : ubFactor_;

        // 换算成切W的逻辑，将loop2用起来作为N上的循环
        uint32_t headLoop{0}, headMode{0}, centerLoop{0}, centerMode{0}, centerBSLoop{1}, tailLoop{0}, tailMode{0},
            wLoop{0};
        uint32_t wSize = tdPtr_->input.outShape[ubAxis_ + 1];
        GetAxisLoop(ubAxis_ + 1, wSize, headLoop, headMode, centerLoop, centerMode, centerBSLoop, tailLoop, tailMode,
                    wLoop);

        uint32_t strideC = tdPtr_->input.outShape[0];
        for (uint8_t i = 0; i < rank_; i++) {
            if (i != 0) {
                strideC *= tdPtr_->input.inShape[i];
            }
        }

        // translate idx(in N) to widx(in W)
        outIndex[0] = idx * ubFactor_; // 每个核上处理ubAxisInCopyNum个N
        uint64_t loopOutIndex[3] = {outIndex[0], 0, 0}; // 原始outIndex用于多次输入后的合并输出，不要污染而启用新变量
        uint32_t W = 0;
        uint64_t wBlockShape = tdPtr_->input.blockShape[blockShapeSize_ - 1];
        DataCopyExtParams copyInParams;
        copyInParams.blockLen = originC_ * sizeof(T);
        copyInParams.srcStride = (strideC - originC_) * sizeof(T);
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = originC_ * sizeof(T);

        for (uint32_t widx = 0; widx < wLoop; widx++) {
            uint32_t wIndex = widx % wLoop;
            uint32_t ubWInCopyNum = 0;
            loopOutIndex[1] = W;
            GetCopyInNum(ubWInCopyNum, wIndex, headLoop, headMode, centerLoop, centerMode, tailLoop, tailMode,
                         centerBSLoop, wSize);
            uint64_t ubInOffset = W * alignC_;
            W += ubWInCopyNum;
            Conver2InIndex(static_cast<uint64_t*>(loopOutIndex), inIndex);
            uint64_t inAddr = 0;
            for (uint8_t i = 0; i < rank_; i++) {
                inAddr += inIndex[i] * inStride_[i];
            }

            bool cond = ubWInCopyNum <= wBlockShape;
            copyInParams.blockCount = cond ? ubWInCopyNum : wBlockShape;
            loopParams.loop1Size = cond ? 1 : ubWInCopyNum / wBlockShape;
            loopParams.loop1DstStride = loopParams.loop1Size == 1 ? 0 : alignC_ * copyInParams.blockCount * sizeof(T);
            loopParams.loop1SrcStride = loopParams.loop1Size == 1 ? 0 : originC_ * sizeof(T);
            loopParams.loop2Size = ubAxisInCopyNum;
            loopParams.loop2SrcStride = inStride_[ubAxis_] * sizeof(T);
            loopParams.loop2DstStride = wSize * alignC_ * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[ubInOffset], inputGm_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
    }

    __aicore__ inline void GetCopyInNum(uint32_t& ubAxisInCopyNum, uint32_t wIndex, uint32_t headLoop,
                                        uint32_t headMode, int32_t centerLoop, uint32_t centerMode, uint32_t tailLoop,
                                        uint32_t tailMode, uint32_t centerBSLoop, uint32_t ubFactor)
    {
        if (wIndex < headLoop) {
            ubAxisInCopyNum = (headMode != 0 && wIndex == headLoop - 1) ? headMode : ubFactor;
        } else if (wIndex >= headLoop && wIndex < headLoop + centerLoop) {
            wIndex -= headLoop;
            if (centerMode != 0) {
                if (centerBSLoop > 1) {
                    if ((wIndex + 1) % (centerLoop / centerBSLoop) == 0) {
                        ubAxisInCopyNum = centerMode;
                        return;
                    }
                } else {
                    if ((wIndex + 1) == centerLoop) {
                        ubAxisInCopyNum = centerMode;
                        return;
                    }
                }
            }
            ubAxisInCopyNum = ubFactor;
        } else {
            wIndex -= headLoop + centerLoop;
            ubAxisInCopyNum = (tailMode != 0 && wIndex == tailLoop - 1) ? tailMode : ubFactor;
        }
    }

    __aicore__ inline void DoCopyInAxisH(const LocalTensor<T>& src, uint64_t* curOutIndex, uint64_t* inIndex,
                                         uint32_t& ubAxisInCopyNum, uint32_t idx)
    {
        uint32_t curUbFactor = getTileNumByIdx(idx, tileQueryNumsDim_, innerTileNumsDim_, headLenH_, tailLenH_,
                                               ubRealFactorTotal_);
        ubAxisInCopyNum = curUbFactor;
        DoCopyInAxisHForTile(src, curOutIndex, inIndex, curUbFactor);
    }

    __aicore__ inline void DoCopyInAxisHForTile(const LocalTensor<T>& src, uint64_t* outIndex, uint64_t* inIndex,
                                                uint32_t curUbFactor)
    {
        if (curUbFactor == 0)
            return;

        uint64_t instrideBSW = instrideBS[rank_ - AXIS_OFFSET_W];
        uint64_t instrideBSH = instrideBS[rank_ - AXIS_OFFSET_H];

        uint64_t leftCopyLen = leftCopyLen_;
        uint64_t rightCopyLen = rightCopyLen_;
        uint64_t middleCopyLen = middleCopyLen_;

        uint32_t ubAddr = 0;

        // 左残块拷贝
        if (leftCopyLen > 0) {
            CopySegment(src, outIndex, inIndex, leftCopyLen, ubAddr, curUbFactor, inW_, inC_, outW_, BSW_, instrideBSW,
                        instrideBSH, true, true);
        }

        // 中间连续段拷贝
        if (middleCopyLen > 0) {
            CopySegment(src, outIndex, inIndex, middleCopyLen, ubAddr, curUbFactor, inW_, inC_, outW_, BSW_,
                        instrideBSW, instrideBSH, false, true);
        }

        // 右残块拷贝
        if (rightCopyLen > 0) {
            CopySegment(src, outIndex, inIndex, rightCopyLen, ubAddr, curUbFactor, inW_, inC_, outW_, BSW_, instrideBSW,
                        instrideBSH, true, false);
        }

        outIndex[rank_ - AXIS_OFFSET_W] = 0;
        updateOutIndexByCarry(outIndex, curUbFactor, rank_ - AXIS_OFFSET_H);
    }

    __aicore__ inline void CalcBoundaryBlock(uint64_t cropStart, uint64_t cropEnd, uint64_t blockSize, uint64_t outLen,
                                             uint64_t& outStartLen, uint64_t& outEndLen, uint64_t& outMiddleLen)
    {
        outStartLen = remainToBoundary(cropStart, blockSize);
        outEndLen = remainToBoundary(cropEnd, blockSize);
        if (outStartLen + outEndLen > outLen) {
            if (cropStart % blockSize != 0) {
                outStartLen = outLen;
                outEndLen = 0;
            } else {
                outStartLen = 0;
                outEndLen = outLen;
            }
            outMiddleLen = 0;
        } else {
            outMiddleLen = safeSubU64(outLen, outStartLen + outEndLen);
        }
    }

    __aicore__ inline uint64_t getTileNumPrefixSum(uint32_t m)
    {
        if (m == 0)
            return 0;

        const uint32_t P = tileQueryNumsDim_;
        if (P == 0)
            return 0;

        uint32_t k = m / P;
        uint32_t r = m % P;

        uint64_t rem = 0;
        for (uint32_t i = 0; i < r; ++i) {
            rem += getTileNumByIdx(i, P, innerTileNumsDim_, headLenH_, tailLenH_, ubRealFactorTotal_);
        }

        return (uint64_t)k * outH_ + rem;
    }

    __aicore__ inline uint32_t getTileNumByInnerModIdx(
        uint32_t innerModIdx, uint32_t ubRealFactorTotal, uint32_t ubFactor,
        bool smallFirst) // true: 小块在前(如15->7,8), false: 小块在后(如15->8,7)
    {
        if (ubFactor == 0 || ubRealFactorTotal == 0) {
            return 0;
        }

        uint32_t fullBlocks = ubRealFactorTotal / ubFactor;
        uint32_t remainder = ubRealFactorTotal % ubFactor;

        if (!smallFirst) {
            // 大块在前，尾小块在后
            if (innerModIdx < fullBlocks) {
                return ubFactor;
            }
            if (innerModIdx == fullBlocks && remainder != 0) {
                return remainder;
            }
            return 0;
        } else {
            // 小块在前，大块在后
            if (remainder != 0) {
                if (innerModIdx == 0) {
                    return remainder;
                }
                if (innerModIdx >= 1 && innerModIdx <= fullBlocks) {
                    return ubFactor;
                }
                return 0;
            } else {
                // 整除时全部是大块
                if (innerModIdx < fullBlocks) {
                    return ubFactor;
                }
                return 0;
            }
        }
    }

    __aicore__ inline uint32_t getTileNumByIdx(uint32_t idx, uint32_t tileQueryNumsDim, uint32_t innerTileNumsDim,
                                               uint32_t headLen, uint32_t tailLen, uint32_t ubRealFactorTotal)
    {
        if (tileQueryNumsDim == 0) {
            return 0;
        }

        uint32_t modIdx = idx % tileQueryNumsDim;

        // 头部区域：使用 smallFirst=true（满足你说的 idx=0 返回7 这种场景）
        if (modIdx < headTilesDim_) {
            return getTileNumByInnerModIdx(modIdx, // 0,1,2...
                                           headLen, ubFactor_, true);
        }

        // 尾部区域：保持原先方向 smallFirst=false
        if (modIdx >= tileQueryNumsDim - tailTilesDim_) {
            uint32_t tailIdx = modIdx - (tileQueryNumsDim - tailTilesDim_);
            return getTileNumByInnerModIdx(tailIdx, // 0,1,2...
                                           tailLen, ubFactor_, false);
        }

        // 中间区域：smallFirst=false
        if (innerTileNumsDim == 0) {
            return 0;
        }

        uint32_t middleIdx = modIdx - headTilesDim_;
        uint32_t innerModIdx = middleIdx % innerTileNumsDim;

        return getTileNumByInnerModIdx(innerModIdx, ubRealFactorTotal, ubFactor_, false);
    }

    __aicore__ inline uint64_t remainToBoundary(uint64_t crop, uint64_t block)
    {
        if (block == 0)
            return 0;
        return (block - (crop % block)) % block;
    }

    __aicore__ inline uint64_t safeSubU64(uint64_t a, uint64_t b) { return (a >= b) ? (a - b) : 0; }

    __aicore__ inline void updateOutIndexByCarry(uint64_t* outIndex, uint32_t curUbFactor, uint32_t axis)
    {
        outIndex[axis] += curUbFactor;
        while (axis < rank_ && tdPtr_->input.outShape[axis] > 0 && outIndex[axis] >= tdPtr_->input.outShape[axis]) {
            outIndex[axis] = 0;
            if (axis > 0) {
                axis--;
                outIndex[axis] += 1;
            } else {
                break;
            }
        }
    }

    __aicore__ inline bool convert(uint64_t* outIndex, uint64_t sum)
    {
        // 1) 清零
        for (uint32_t i = 0; i < MAX_INPUT_RANK; ++i) {
            outIndex[i] = 0;
        }

        uint32_t axis = rank_ - AXIS_OFFSET_H; // 例如 rank=4 -> axis=1

        outIndex[axis] = sum;

        // 3) 从 axis 往前连续进位到 0 轴
        for (int32_t d = (int32_t)axis; d > 0; --d) {
            uint64_t dim = tdPtr_->input.outShape[d];
            if (dim == 0)
                return false;

            uint64_t carry = outIndex[d] / dim;
            outIndex[d] = outIndex[d] % dim;
            outIndex[d - 1] += carry;
        }

        // 4) 检查最高维是否越界
        if (outIndex[0] >= tdPtr_->input.outShape[0]) {
            return false;
        }

        return true;
    }

    __aicore__ inline void CopySegment(const LocalTensor<T>& src, uint64_t* outIndex, uint64_t* inIndex,
                                       uint64_t copyLen, uint32_t& ubAddr, uint32_t curUbFactor, uint64_t inW,
                                       uint64_t inC, uint64_t outW, uint64_t BSW, uint64_t instrideBSW,
                                       uint64_t instrideBSH, bool isResidual, bool updateAddr)
    {
        Conver2InIndex(outIndex, inIndex);

        // 计算GM地址
        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < rank_; i++) {
            inAddr += inIndex[i] * inStride_[i];
        }

        // 配置拷贝参数
        DataCopyExtParams copyParams;
        copyParams.blockLen = inC * sizeof(T);
        copyParams.srcStride = (instrideBSW - inC) * sizeof(T);
        copyParams.dstStride = 0;

        LoopModeParams loopParams;
        DataCopyPadExtParams<T> padParams(false, 0, 0, 0);

        // 设置loop1参数
        if (!isResidual && copyLen > BSW) {
            copyParams.blockCount = BSW;
            loopParams.loop1Size = copyLen / BSW;
            loopParams.loop1SrcStride = inC * sizeof(T);
            loopParams.loop1DstStride = alignedCLength_ * BSW * sizeof(T);
        } else {
            copyParams.blockCount = copyLen;
            loopParams.loop1Size = 1;
            loopParams.loop1SrcStride = 0;
            loopParams.loop1DstStride = 0;
        }

        // 设置loop2参数
        uint32_t outerLoop = 1;
        loopParams.loop2SrcStride = instrideBSH * sizeof(T);
        loopParams.loop2DstStride = alignedCLength_ * outW * sizeof(T);
        if (curUbFactor <= BSH_) {
            loopParams.loop2Size = curUbFactor;
        } else {
            loopParams.loop2Size = BSH_;
            outerLoop = curUbFactor / BSH_;
        }

        // 执行拷贝
        for (uint32_t i = 0; i < outerLoop; i++) {
            uint32_t gmOffset = inW * inC * i;
            uint32_t ubOffset = BSH_ * outW * alignedCLength_ * i;

            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[ubAddr + ubOffset], inputGm_[inAddr + gmOffset], copyParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }

        // 更新地址和索引
        if (updateAddr) {
            ubAddr += copyLen * alignedCLength_;
            outIndex[rank_ - AXIS_OFFSET_W] += copyLen;
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, const uint64_t* outIndex, uint32_t ubAxisOutCopyNum)
    {
        uint64_t outAddr = 0;
        for (uint32_t i = 0; i < rank_; i++) {
            outAddr += outIndex[i] * outStride_[i];
        }
        DataCopyExtParams copyOutParams;
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        if (ubAxis_ == rank_ - 1) {
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = ubAxisOutCopyNum * outStride_[ubAxis_] * sizeof(T);
        } else if (ubAxis_ == rank_ - AXIS_OFFSET_W) {
            copyOutParams.blockCount = ubAxisOutCopyNum;
            copyOutParams.blockLen = originC_ * sizeof(T);
        } else if (ubAxis_ == rank_ - AXIS_OFFSET_H && rank_ == RANK_3D) {
            copyOutParams.blockCount = ubAxisOutCopyNum * tdPtr_->input.outShape[ubAxis_ + 1]; // N-Axis*L
            copyOutParams.blockLen = originC_ * sizeof(T);                                     // C
        } else if (ubAxis_ == rank_ - AXIS_OFFSET_H && rank_ >= MIN_RANK_FOR_H_AXIS) {         // H轴
            copyOutParams.blockCount = ubAxisOutCopyNum * tdPtr_->input.outShape[rank_ - AXIS_OFFSET_W];
            copyOutParams.blockLen = tdPtr_->input.outShape[rank_ - 1] * sizeof(T);
        }
        DataCopyPad(outputGm_[outAddr], src[0], copyOutParams);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }
};
} // namespace B2SND

#endif // BATCH_TO_SPACE_LARGEC_H
