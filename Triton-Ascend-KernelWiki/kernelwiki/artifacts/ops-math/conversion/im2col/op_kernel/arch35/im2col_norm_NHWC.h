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
 * \file im2col_nhwc_normal.h
 * \brief
 */

#ifndef _IM2COL_NORMAL_NHWC_H_
#define _IM2COL_NORMAL_NHWC_H_

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "im2col_tilingdata.h"

namespace Im2col {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, bool isPadding, uint8_t ubAxis>
class KernelIm2ColNormNhwc {
private:
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint32_t UB_BLOCK = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    constexpr static uint8_t MAX_DIMS_NUM = 4;
    constexpr static uint8_t C_AXIS = 3;
    constexpr static uint8_t W_AXIS = 2;
    constexpr static uint8_t H_AXIS = 1;
    constexpr static uint8_t N_AXIS = 0;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    int64_t outC{0};
    int64_t outW{0};
    int64_t outH{0};
    int64_t inC{0};
    int64_t inH{0};
    int64_t inW{0};
    int64_t inStride_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t outStride_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t inShape_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t outShape_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t inIndex_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t outIndex_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t ubFactor_[MAX_DIMS_NUM] = {0, 0, 0, 0};
    int64_t convKernelNumInWidth_{0};
    int64_t convKernelNumInHeight_{0};
    int64_t hStride_{0};
    int64_t wStride_{0};
    int64_t hDilation_{0};
    int64_t wDilation_{0};
    int64_t hPaddingTop_{0};
    int64_t wPaddingTop_{0};
    int64_t wPaddingBottom_{0};
    int64_t hKernelSize_{0};
    int64_t wKernelSize_{0};
    int64_t wKernelEffSize_{0};
    int64_t hKernelEffSize_{0};
    int64_t blockIdx_;
    int64_t ubRealFactor_{0};
    uint32_t alignedCLength_{0};
    int32_t outputBufferSize_{0};
    int32_t outputTileSize_{0};
    bool isWPadding_{false};

    TPipe* pipe_ = nullptr;
    const Im2ColNHWCTilingData* tilingData_ = nullptr;
    TBuf<TPosition::VECCALC> inQueue_;
    T constValue_{0};

public:
    __aicore__ inline KernelIm2ColNormNhwc() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const Im2ColNHWCTilingData* tilingData, TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        input_.SetGlobalBuffer((__gm__ T*)x);
        output_.SetGlobalBuffer((__gm__ T*)y);
        outputBufferSize_ = tilingData_->outputBufferSize;
        outputTileSize_ = tilingData_->outputBufferSize / BUFFER_NUM;
        pipe_->InitBuffer(inQueue_, outputBufferSize_);
        convKernelNumInWidth_ = tilingData_->convKernelNumInWidth;
        convKernelNumInHeight_ = tilingData_->convKernelNumInHeight;
        hStride_ = tilingData_->input.hStride;
        wStride_ = tilingData_->input.wStride;
        hDilation_ = tilingData_->input.hDilation;
        wDilation_ = tilingData_->input.wDilation;
        hPaddingTop_ = tilingData_->input.hPaddingBefore;
        wPaddingTop_ = tilingData_->input.wPaddingBefore;
        wPaddingBottom_ = tilingData_->input.wPaddingAfter;
        hKernelSize_ = tilingData_->input.hKernelSize;
        wKernelSize_ = tilingData_->input.wKernelSize;
        wKernelEffSize_ = (wKernelSize_ - 1) * wDilation_ + 1;
        hKernelEffSize_ = (hKernelSize_ - 1) * hDilation_ + 1;
        inH = tilingData_->input.H;
        inW = tilingData_->input.W;
        inC = outC = tilingData_->input.C;
        outW = hKernelSize_ * wKernelSize_;
        outH = convKernelNumInWidth_ * convKernelNumInHeight_;
        inShape_[N_AXIS] = outShape_[N_AXIS] = tilingData_->input.N;
        inShape_[C_AXIS] = outShape_[C_AXIS] = tilingData_->input.C;
        inShape_[W_AXIS] = tilingData_->input.W;
        inShape_[H_AXIS] = tilingData_->input.H;
        outShape_[W_AXIS] = outW;
        outShape_[H_AXIS] = outH;
        ubFactor_[N_AXIS] = tilingData_->ubFactorN;
        ubFactor_[H_AXIS] = tilingData_->ubFactorH;
        ubFactor_[W_AXIS] = tilingData_->ubFactorW;
        ubFactor_[C_AXIS] = tilingData_->ubFactorC;
        int64_t inShapeSize_ = 1UL;
        int64_t outShapeSize_ = 1UL;
        for (int8_t i = MAX_DIMS_NUM - 1; i >= 0; --i) {
            inStride_[i] = inShapeSize_;
            inShapeSize_ *= inShape_[i];
            outStride_[i] = outShapeSize_;
            outShapeSize_ *= outShape_[i];
        }
        alignedCLength_ = CeilAlign(static_cast<uint32_t>(outShape_[MAX_DIMS_NUM - 1]), BLK_ELEMS);
        isWPadding_ = wPaddingTop_ != 0 || wPaddingBottom_ != 0;
    }

    __aicore__ inline void Process()
    {
        int64_t startIdx = blockIdx_ * tilingData_->linesPerCore;
        if (startIdx >= tilingData_->totalLines) {
            return;
        }
        LocalTensor<T> input = inQueue_.Get<T>();

        // 把首次处理的ub块先用0来填充
        if constexpr (isPadding) {
            Duplicate(input, constValue_, outputBufferSize_ / sizeof(T));
            SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        }

        int64_t endIdx = (blockIdx_ + 1L) * tilingData_->linesPerCore;
        endIdx = (endIdx < tilingData_->totalLines ? endIdx : tilingData_->totalLines);

        for (int64_t idx = startIdx; idx < endIdx; idx++) {
            int64_t curIdx = idx;
            for (int8_t i = ubAxis; i >= 0; i--) {
                CalculateOutIndex(curIdx, i);
            }
            inIndex_[C_AXIS] = outIndex_[C_AXIS];
            inIndex_[N_AXIS] = outIndex_[N_AXIS];
            inIndex_[H_AXIS] = outIndex_[H_AXIS] / convKernelNumInWidth_ * hStride_ -
                hPaddingTop_ + outIndex_[W_AXIS] * hKernelSize_ / outW * hDilation_;
            inIndex_[W_AXIS] = outIndex_[H_AXIS] % convKernelNumInWidth_ * wStride_ - wPaddingTop_ +
                (outIndex_[W_AXIS] - (outIndex_[W_AXIS] * hKernelSize_ / outW) * (outW / hKernelSize_)) * wDilation_;

            LocalTensor<T> srcLocal = input[((idx - startIdx) & (BUFFER_NUM - 1)) * outputTileSize_ / sizeof(T)];
            CopyIn(srcLocal, idx - startIdx);
            SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            CopyOut(srcLocal, idx - startIdx);
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }

    __aicore__ inline void CalculateOutIndex(int64_t& curIdx, int8_t curUbAxis)
    {
        int64_t factorNum = outShape_[curUbAxis];
        if (curUbAxis == ubAxis) {
            if (ubAxis == W_AXIS && ubFactor_[curUbAxis] < wKernelSize_) {
                int64_t wFactorNum = CeilDiv(wKernelSize_, ubFactor_[curUbAxis]);
                factorNum = wFactorNum * hKernelSize_;
                if (wFactorNum != 0) {
                    ubRealFactor_ = (curIdx % wFactorNum + 1) * ubFactor_[curUbAxis] <= wKernelSize_ ?
                        ubFactor_[curUbAxis] : wKernelSize_ % ubFactor_[curUbAxis];
                    outIndex_[curUbAxis] = ((curIdx % wFactorNum) * ubFactor_[curUbAxis] +
                        wKernelSize_ * (curIdx / wFactorNum)) % outShape_[curUbAxis];
                }
            } else if (ubAxis == H_AXIS && ubFactor_[curUbAxis] < convKernelNumInWidth_) {
                int64_t hFactorNum = CeilDiv(convKernelNumInWidth_, ubFactor_[curUbAxis]);
                factorNum = hFactorNum * convKernelNumInHeight_;
                if (hFactorNum != 0) {
                    ubRealFactor_ = (curIdx % hFactorNum + 1) * ubFactor_[curUbAxis] <= convKernelNumInWidth_ ?
                        ubFactor_[curUbAxis] : convKernelNumInWidth_ % ubFactor_[curUbAxis];
                    outIndex_[curUbAxis] = ((curIdx % hFactorNum) * ubFactor_[curUbAxis] +
                        convKernelNumInWidth_ * (curIdx / hFactorNum)) % outShape_[curUbAxis];
                }
            } else {
                factorNum = CeilDiv(outShape_[curUbAxis], ubFactor_[curUbAxis]);
                if (factorNum != 0) {
                    outIndex_[curUbAxis] = curIdx % factorNum * ubFactor_[curUbAxis];
                }
                ubRealFactor_ = Std::min(ubFactor_[curUbAxis], outShape_[curUbAxis] - outIndex_[curUbAxis]);
            }
        } else {
            outIndex_[curUbAxis] = curIdx % factorNum;
        }
        curIdx = (factorNum != 0) ? curIdx / factorNum : curIdx;
    }

    __aicore__ inline void CopyIn(const LocalTensor<T>& src, int32_t idx)
    {
        if (idx >= 1) {
            if constexpr (isPadding) {
                int32_t nextIdx = idx + 1;
                LocalTensor<T> input = inQueue_.Get<T>();
                LocalTensor<T> nextLocal =
                    input[(nextIdx & (BUFFER_NUM - 1)) * outputTileSize_ / sizeof(T)];
                Duplicate(nextLocal, constValue_, outputTileSize_ / sizeof(T));
            }
            SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        }

        if constexpr (ubAxis == C_AXIS) {
            DoCopyInAxisC(src);
        }
        if constexpr (ubAxis == W_AXIS) {
            DoCopyInAxisW(src);
        }
        if constexpr (ubAxis == H_AXIS) {
            DoCopyInAxisH(src, ubRealFactor_);
        }
        if constexpr (ubAxis == N_AXIS) {
            DoCopyInAxisN(src);
        }
    }

    __aicore__ inline void DoCopyInAxisC(const LocalTensor<T>& src)
    {
        bool isAllPadding = inIndex_[H_AXIS] >= 0 && inIndex_[H_AXIS] < inH &&
            inIndex_[W_AXIS] >= 0 && inIndex_[W_AXIS] < inW ? false : true;
        if (isAllPadding) {
            return;
        }
        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < MAX_DIMS_NUM; i++) {
            inAddr += inIndex_[i] * inStride_[i];
        }
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        copyInParams.blockCount = 1;
        copyInParams.blockLen = ubRealFactor_ * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(src, input_[inAddr], copyInParams, padParams);
    }

    __aicore__ inline void DoCopyInAxisW(const LocalTensor<T>& src)
    {
        int64_t inHLast = inIndex_[H_AXIS] + (CeilDiv(ubRealFactor_, wKernelSize_) - 1) * hDilation_;
        int64_t inWLast = inIndex_[W_AXIS] + (Std::min(ubRealFactor_, wKernelSize_) - 1) * wDilation_;
        int64_t startValidHIndex = inIndex_[H_AXIS] + CeilDiv(Std::max(
            0L, inIndex_[H_AXIS]) - inIndex_[H_AXIS], hDilation_) * hDilation_;
        int64_t startValidWIndex = inIndex_[W_AXIS] + CeilDiv(Std::max(
            0L, inIndex_[W_AXIS]) - inIndex_[W_AXIS], wDilation_) * wDilation_;
        if (inIndex_[H_AXIS] >= inH || inHLast < 0 || inIndex_[W_AXIS] >= inW || inWLast < 0 ||
            startValidWIndex < 0 || startValidWIndex > inWLast ||
            startValidHIndex < 0 || startValidHIndex > inHLast) { // 无有效采样点，直接返回
            return;
        }

        // 计算出最后一个有效采样点所在的H索引和W索引
        int64_t hBound = inHLast >= inH ? inH - 1 : inHLast;
        int64_t endValidHIndex = inIndex_[H_AXIS] + (hBound - inIndex_[H_AXIS]) / hDilation_ * hDilation_;
        int64_t wBound = inWLast >= inW ? inW - 1 : inWLast;
        int64_t endValidWIndex = inIndex_[W_AXIS] + (wBound - inIndex_[W_AXIS]) / wDilation_ * wDilation_;

        // 计算ub偏移，pad部分是统一使用duplicate设置的，本次搬运要跳过pad部分的长度
        uint32_t ubInOffset = ((startValidHIndex - inIndex_[H_AXIS]) / hDilation_ * wKernelSize_ +
            (startValidWIndex - inIndex_[W_AXIS]) / wDilation_) * alignedCLength_;

        // 更新去除前pad后首个需要拷贝数据的坐标位置以及计算inAddr
        inIndex_[H_AXIS] = startValidHIndex;
        inIndex_[W_AXIS] = startValidWIndex;
        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < MAX_DIMS_NUM; i++) {
            inAddr += inIndex_[i] * inStride_[i];
        }
        // 初始化搬运参数
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        copyInParams.blockCount = (endValidWIndex - startValidWIndex) / wDilation_ + 1;
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = (endValidHIndex - startValidHIndex) / hDilation_ + 1;
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(src[ubInOffset], input_[inAddr], copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void DoCopyInAxisH(const LocalTensor<T>& src, int64_t ubFactorH)
    {
        if constexpr (isPadding) {
            DoCopyInAxisHWithPad(src, ubFactorH);
        } else {
            DoCopyInAxisHWithoutPad(src, ubFactorH);
        }
    }

    __aicore__ inline void DoCopyInAxisHWithoutPad(const LocalTensor<T>& src, int64_t ubFactorH)
    {
        // 计算需要处理H方向的滑动次数
        uint32_t hSlideNum = CeilDiv(ubFactorH, convKernelNumInWidth_);

        // 计算gm偏移
        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < MAX_DIMS_NUM; ++i) {
            inAddr += inIndex_[i] * inStride_[i];
        }

        // 计算当前坐标在w方向上一共滑动出有多少个（包括首次，同时需要和ubFactorH取最小）
        uint32_t slideNum = Std::min(ubFactorH,
            (inW + wPaddingBottom_ - 1 -(inIndex_[W_AXIS] + wKernelEffSize_ - 1)) / wStride_ + 1);

        // 初始化搬运参数
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;
        copyInParams.blockCount = wKernelSize_;
        LoopModeParams loopParams;
        loopParams.loop1Size = hKernelSize_;
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);

        // W方向滑动次数不小于H方向场景的搬运
        if (slideNum >= hSlideNum) {
            loopParams.loop2Size = slideNum;
            loopParams.loop2SrcStride = inC * wStride_ * sizeof(T);
            loopParams.loop2DstStride = outW * alignedCLength_ * sizeof(T);
            for (uint32_t i = 0; i < hSlideNum; ++i) {
                uint32_t gmInOffset = inW * inC * hStride_ * i;
                uint32_t ubInOffset = i * slideNum * outW * alignedCLength_;
                SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                DataCopyPad(src[ubInOffset], input_[inAddr + gmInOffset], copyInParams, padParams);
                ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            }
            return;
        }

        // H方向滑动次数大于W方向场景的搬运
        loopParams.loop2Size = hSlideNum;
        loopParams.loop2SrcStride = inW * inC * hStride_ * sizeof(T);
        loopParams.loop2DstStride = outW * slideNum * alignedCLength_ * sizeof(T);
        
        for (uint32_t i = 0; i < slideNum; ++i) {
            uint32_t gmInOffset = inC * wStride_ * i;
            uint32_t ubInOffset = outW * alignedCLength_ * i;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[ubInOffset], input_[inAddr + gmInOffset], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
    }

    __aicore__ inline void DoCopyInAxisHWithPad(const LocalTensor<T>& src, int64_t ubFactorH)
    {
        uint32_t hSlideNum = CeilDiv(ubFactorH, convKernelNumInWidth_);
        uint32_t wSlideNum = Std::min(ubFactorH,
            (inW + wPaddingBottom_ - 1 -(inIndex_[W_AXIS] + wKernelEffSize_ - 1)) / wStride_ + 1);
        // W方向有效滑动次数多，优化将W方向的有效滑动次数作为loop参数
        if (hSlideNum == 1 || wSlideNum >= hSlideNum) {
            DoCopyInAxisConvWPrefer(hSlideNum, wSlideNum, src);
            return;
        }

        // 将H方向的有效滑动次数作为loop参数
        DoCopyInAxisConvHPrefer(hSlideNum, wSlideNum, src);
    }

    __aicore__ inline void DoCopyInAxisConvWPrefer(uint32_t hSlideNum, uint32_t wSlideNum, const LocalTensor<T>& src)
    {
        for (uint32_t i = 0; i < hSlideNum; ++i) {
            int64_t inHLast = inIndex_[H_AXIS] + (hKernelSize_ - 1) * hDilation_;
            int64_t startValidHIndex = inIndex_[H_AXIS] + CeilDiv(Std::max(
                0L, inIndex_[H_AXIS]) - inIndex_[H_AXIS], hDilation_) * hDilation_;
            int64_t endValidHIndex = inIndex_[H_AXIS] + (Std::min(inHLast, inH - 1) - inIndex_[H_AXIS]) / hDilation_ * hDilation_;
            // h没有落在有效范围内
            if (inIndex_[H_AXIS] >= inH || inHLast < 0 ||
                startValidHIndex < 0 || startValidHIndex > inHLast || endValidHIndex < 0) { // h没有落在有效范围内
                inIndex_[H_AXIS] += hStride_;
                continue;
            }

            int64_t oriWAxis = inIndex_[W_AXIS];
            uint32_t j = 0;
            while (j < wSlideNum) {
                int64_t inWLast = inIndex_[W_AXIS] + (wKernelSize_ - 1) * wDilation_;
                uint64_t ubStartAddr = (i * wSlideNum + j) * outW * alignedCLength_;

                // 处理卷积采样点有pad区域的情况
                if (inIndex_[W_AXIS] < 0 || inWLast >= inW) {
                    DoCopyInKernelWSlideWithPad(startValidHIndex, endValidHIndex, inWLast, src, ubStartAddr);
                    inIndex_[W_AXIS] += wStride_;
                    ++j;
                    continue;
                }

                // 处理卷积采样点全部落在原始图像（非pad）的情况
                uint32_t validSlideNum = (inW - (inIndex_[W_AXIS] + wKernelEffSize_)) / wStride_ + 1;
                uint32_t untreatedWSlideNum = Std::min(validSlideNum, wSlideNum - j);
                DoCopyInKernelWSlideWithoutPad(startValidHIndex, endValidHIndex, untreatedWSlideNum, src, ubStartAddr);
                inIndex_[W_AXIS] += wStride_ * untreatedWSlideNum;
                j += untreatedWSlideNum;
            }
            inIndex_[H_AXIS] += hStride_;
            inIndex_[W_AXIS] = oriWAxis;
        }
    }

    __aicore__ inline void DoCopyInAxisConvHPrefer(
        const uint32_t hSlideNum, const uint32_t wSlideNum, const LocalTensor<T>& src)
    {
        uint32_t h = 0;
        while (h < hSlideNum) {
            int64_t inHLast = inIndex_[H_AXIS] + (hKernelSize_ - 1) * hDilation_;
            int64_t startValidHIndex = inIndex_[H_AXIS] + CeilDiv(Std::max(
                0L, inIndex_[H_AXIS]) - inIndex_[H_AXIS], hDilation_) * hDilation_;
            int64_t endValidHIndex = inIndex_[H_AXIS] + (Std::min(inHLast, inH - 1) - inIndex_[H_AXIS]) / hDilation_ * hDilation_;    
            if (inIndex_[H_AXIS] >= inH || inHLast < 0 ||
                startValidHIndex < 0 || startValidHIndex > inHLast || endValidHIndex < 0) {
                inIndex_[H_AXIS] += hStride_;
                ++h;
                continue;
            }
            int64_t oriWAxis = inIndex_[W_AXIS];
            uint32_t ubStartAddr = h * wSlideNum * outW * alignedCLength_;

            // 处理kernel采样点全部在有效H的场景
            if (inIndex_[H_AXIS] >= 0 && inHLast < inH) {
                uint32_t theHSlideNum = (inH - (inIndex_[H_AXIS] + hKernelEffSize_)) / hStride_ + 1;
                uint32_t untreatedHSlideNum = Std::min(theHSlideNum, hSlideNum - h);
                DoCopyInKernelHSlideWithoutPad(endValidHIndex - startValidHIndex, untreatedHSlideNum, wSlideNum, src, ubStartAddr);
                inIndex_[H_AXIS] += hStride_ * untreatedHSlideNum;
                inIndex_[W_AXIS] = oriWAxis;
                h += untreatedHSlideNum;
                continue;
            }

            uint32_t w = 0;
            while (w < wSlideNum) {
                ubStartAddr = outW * wSlideNum * h * alignedCLength_ + w * outW * alignedCLength_;
                int64_t inWLast = inIndex_[W_AXIS] + (wKernelSize_ - 1) * wDilation_;
                if (isWPadding_) {
                    DoCopyInKernelWSlideWithPad(startValidHIndex, endValidHIndex, inWLast, src, ubStartAddr);
                    inIndex_[W_AXIS] += wStride_;
                    ++w;
                } else {
                    DoCopyInKernelWSlideWithoutPad(startValidHIndex, endValidHIndex, convKernelNumInWidth_ , src, ubStartAddr);
                    inIndex_[W_AXIS] += convKernelNumInWidth_ * wStride_;
                    w += convKernelNumInWidth_;
                }
            }
            inIndex_[H_AXIS] += hStride_;
            inIndex_[W_AXIS] = oriWAxis;
            ++h;
        }
    }

    __aicore__ inline void DoCopyInKernelHSlideWithoutPad(const int64_t validHLength,
        const uint32_t untreatedHSlideNum, const uint32_t wSlideNum, const LocalTensor<T>& src, uint32_t ubStartAddr)
    {
        DataCopyExtParams copyInParams;
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;
        LoopModeParams loopParams;
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop2Size = untreatedHSlideNum;
        loopParams.loop2SrcStride = inW * inC * hStride_ * sizeof(T);
        loopParams.loop2DstStride = outW * wSlideNum * alignedCLength_ * sizeof(T);
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        for (uint32_t w = 0; w < wSlideNum; ++w) {
            int64_t startValidWIndex = inIndex_[W_AXIS] + CeilDiv(Std::max(
                0L, inIndex_[W_AXIS]) - inIndex_[W_AXIS], wDilation_) * wDilation_;
            int64_t inWLast = inIndex_[W_AXIS] + (wKernelSize_ - 1) * wDilation_;    
            int64_t endValidWIndex = inIndex_[W_AXIS] + (Std::min(inWLast, inW - 1) - inIndex_[W_AXIS]) / wDilation_ * wDilation_;
            if (inIndex_[W_AXIS] >= inW || inWLast < 0 || startValidWIndex < 0 ||
                startValidWIndex > inWLast || endValidWIndex < 0) {
                inIndex_[W_AXIS] += wStride_;
                continue;
            }
            uint64_t inAddr = startValidWIndex * inStride_[W_AXIS];
            for (uint8_t i = 0; i < MAX_DIMS_NUM; ++i) {
                if (i == W_AXIS) {
                    continue;
                }
                inAddr += inIndex_[i] * inStride_[i];
            }
            uint32_t ubInOffset = outW * alignedCLength_ * w +
                (startValidWIndex - inIndex_[W_AXIS]) / wDilation_ * alignedCLength_;
            copyInParams.blockCount = (endValidWIndex - startValidWIndex) / wDilation_ + 1;
            loopParams.loop1Size = validHLength / hDilation_ + 1;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(src[ubStartAddr + ubInOffset], input_[inAddr], copyInParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            inIndex_[W_AXIS] += wStride_;
        }
    }

    __aicore__ inline void DoCopyInKernelWSlideWithPad(const int64_t startValidHIndex,
        const int64_t endValidHIndex, const int64_t inWLast, const LocalTensor<T>& src, uint32_t ubStartAddr)
    {
        int64_t startValidWIndex = inIndex_[W_AXIS] + CeilDiv(Std::max(
            0L, inIndex_[W_AXIS]) - inIndex_[W_AXIS], wDilation_) * wDilation_;
        int64_t endValidWIndex = inIndex_[W_AXIS] + (Std::min(inWLast, inW - 1) - inIndex_[W_AXIS]) / wDilation_ * wDilation_;
        // w没有落在有效范围内
        if (inIndex_[W_AXIS] >= inW || inWLast < 0 || startValidWIndex < 0 ||
            startValidWIndex > inWLast || endValidWIndex < 0) { // w没有落在有效范围内
            return;
        }

        // 更新去除前pad后首个需要拷贝数据的坐标位置
        uint64_t inAddr = startValidHIndex * inStride_[H_AXIS] + startValidWIndex * inStride_[W_AXIS];
        for (uint8_t dim = 0; dim < MAX_DIMS_NUM; dim += C_AXIS) {
            inAddr += inIndex_[dim] * inStride_[dim];
        }

        // 初始化搬运参数
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = (endValidWIndex - startValidWIndex) / wDilation_ + 1;
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;
        LoopModeParams loopParams;
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop1Size = (endValidHIndex - startValidHIndex) / hDilation_ + 1;
        loopParams.loop2Size = 1;
        uint32_t ubInOffset = ((startValidHIndex - inIndex_[H_AXIS]) / hDilation_ * wKernelSize_ +
            (startValidWIndex - inIndex_[W_AXIS]) / wDilation_) * alignedCLength_;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(src[ubStartAddr + ubInOffset], input_[inAddr], copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void DoCopyInKernelWSlideWithoutPad(const int64_t startValidHIndex,
        const int64_t endValidHIndex, const uint32_t untreatedWSlideNum, const LocalTensor<T>& src, uint32_t ubStartAddr)
    {
        // 初始化搬运参数
        DataCopyExtParams copyInParams;
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.blockCount = wKernelSize_;
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;

        LoopModeParams loopParams;
        loopParams.loop1Size = (endValidHIndex - startValidHIndex) / hDilation_ + 1;
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop2Size = untreatedWSlideNum;
        loopParams.loop2SrcStride = inC * wStride_ * sizeof(T);
        loopParams.loop2DstStride = outW * alignedCLength_ * sizeof(T);

        uint64_t inAddr = startValidHIndex * inStride_[H_AXIS];
        for (uint8_t dim = 0; dim < MAX_DIMS_NUM; ++dim) {
            if (dim == H_AXIS) {
                continue;
            }
            inAddr += inIndex_[dim] * inStride_[dim];
        }
        uint32_t ubInOffset = ((startValidHIndex - inIndex_[H_AXIS]) / hDilation_ * wKernelSize_) * alignedCLength_;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(src[ubStartAddr + ubInOffset], input_[inAddr], copyInParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void DoCopyInAxisN(const LocalTensor<T>& src)
    {
        // ubFactorN比W和H方向的滑动窗口数量多，将ubFactorN作为loop的参数
        if (ubRealFactor_ > convKernelNumInWidth_ && ubRealFactor_ > convKernelNumInHeight_) {
            if constexpr (isPadding) {
                DoCopyInAxisNWithPad(src);
            } else {
                DoCopyInAxisNWithoutPad(src);
            }
        } else { // N比W或者H方向的滑动窗口少
            int64_t curHAxis = inIndex_[H_AXIS];
            for (uint32_t i = 0; i < ubRealFactor_; ++i) {
                DoCopyInAxisH(src[outW * outH * alignedCLength_ * i], ubFactor_[H_AXIS]);
                inIndex_[N_AXIS]++;
                inIndex_[H_AXIS] = curHAxis;
            }
        }
    }

    __aicore__ inline void DoCopyInAxisNWithPad(const LocalTensor<T>& src)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;
        LoopModeParams loopParams;
        loopParams.loop2Size = ubRealFactor_;
        loopParams.loop2SrcStride = inStride_[N_AXIS] * sizeof(T);
        loopParams.loop2DstStride = outW * outH * alignedCLength_ * sizeof(T);
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);
        for (uint32_t h = 0; h < convKernelNumInHeight_; ++h) {
            int64_t inHLast = inIndex_[H_AXIS] + (hKernelSize_ - 1) * hDilation_;
            int64_t startValidHIndex = inIndex_[H_AXIS] + CeilDiv(Std::max(
                0L, inIndex_[H_AXIS]) - inIndex_[H_AXIS], hDilation_) * hDilation_;
            int64_t endValidHIndex = inIndex_[H_AXIS] + (Std::min(inHLast, inH - 1) - inIndex_[H_AXIS]) / hDilation_ * hDilation_;
            if (inIndex_[H_AXIS] >= inH || inHLast < 0 ||
                startValidHIndex < 0 || startValidHIndex > inHLast || endValidHIndex < 0) {
                inIndex_[H_AXIS] += hStride_;
                continue;
            }
            loopParams.loop1Size = (endValidHIndex - startValidHIndex) / hDilation_ + 1;
            int64_t oriWAxis = inIndex_[W_AXIS];
            for (uint32_t w = 0; w < convKernelNumInWidth_; ++w) {
                int64_t inWLast = inIndex_[W_AXIS] + (wKernelSize_ - 1) * wDilation_;
                int64_t startValidWIndex = inIndex_[W_AXIS] + CeilDiv(Std::max(
                    0L, inIndex_[W_AXIS]) - inIndex_[W_AXIS], wDilation_) * wDilation_;
                int64_t endValidWIndex = inIndex_[W_AXIS] + (Std::min(inWLast, inW - 1) - inIndex_[W_AXIS]) / wDilation_ * wDilation_;    
                if (inIndex_[W_AXIS] >= inW || inWLast < 0 ||
                    startValidWIndex < 0 || startValidWIndex > inWLast || endValidWIndex < 0) {
                    inIndex_[W_AXIS] += wStride_;
                    continue;
                }
                copyInParams.blockCount = (endValidWIndex - startValidWIndex) / wDilation_ + 1;
                uint64_t inAddr = startValidHIndex * inStride_[H_AXIS] + startValidWIndex * inStride_[W_AXIS];
                for (uint8_t dim = 0; dim < MAX_DIMS_NUM; dim += C_AXIS) {
                    inAddr += inIndex_[dim] * inStride_[dim];
                }
                uint32_t ubInOffset = outW * alignedCLength_ * (w + convKernelNumInWidth_ * h) + alignedCLength_ *
                    ((startValidHIndex - inIndex_[H_AXIS]) / hDilation_ * wKernelSize_ +
                    (startValidWIndex - inIndex_[W_AXIS]) / wDilation_);
                SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                DataCopyPad(src[ubInOffset], input_[inAddr], copyInParams, padParams);
                ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
                inIndex_[W_AXIS] += wStride_;
            }
            inIndex_[H_AXIS] += hStride_;
            inIndex_[W_AXIS] = oriWAxis;
        }
    }

    __aicore__ inline void DoCopyInAxisNWithoutPad(const LocalTensor<T>& src)
    {
        // 计算gm偏移
        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < MAX_DIMS_NUM; ++i) {
            inAddr += inIndex_[i] * inStride_[i];
        }

        // 初始化搬运参数
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        copyInParams.blockLen = inC * sizeof(T);
        copyInParams.srcStride = (wDilation_ - 1) * inC * sizeof(T);
        copyInParams.dstStride = 0;
        copyInParams.blockCount = wKernelSize_;
        LoopModeParams loopParams;
        loopParams.loop2Size = ubRealFactor_;
        loopParams.loop2SrcStride = inStride_[N_AXIS] * sizeof(T);
        loopParams.loop2DstStride = outW * outH * alignedCLength_ * sizeof(T);
        loopParams.loop1Size = hKernelSize_;
        loopParams.loop1SrcStride = inW * inC * hDilation_ * sizeof(T);
        loopParams.loop1DstStride = wKernelSize_ * alignedCLength_ * sizeof(T);
        for (uint32_t h = 0; h < convKernelNumInHeight_; ++h) {
            for (uint32_t w = 0; w < convKernelNumInWidth_; ++w) {
                uint32_t gmInOffset = inC * wStride_ * w + inW * inC * hStride_ * h;
                uint32_t ubInOffset = outW * alignedCLength_ * w + outW * alignedCLength_ * convKernelNumInWidth_ * h;
                SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                DataCopyPad(src[ubInOffset], input_[inAddr + gmInOffset], copyInParams, padParams);
                ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            }
        }
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& src, const int32_t idx)
    {
        uint64_t outAddr = 0;
        for (uint8_t i = 0; i < MAX_DIMS_NUM; ++i) {
            outAddr += outIndex_[i] * outStride_[i];
        }

        DataCopyExtParams copyOutParams;
        copyOutParams.dstStride = 0;
        if constexpr (ubAxis == C_AXIS) {
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = ubRealFactor_ * sizeof(T);
            copyOutParams.srcStride = 0;
        } else {
            copyOutParams.blockLen = outShape_[C_AXIS] * sizeof(T);
            copyOutParams.srcStride = 0;
            if constexpr (ubAxis == W_AXIS) {
                copyOutParams.blockCount = ubRealFactor_;
            }
            if constexpr (ubAxis == H_AXIS) {
                copyOutParams.blockCount = ubRealFactor_ * outW;
            }
            if constexpr (ubAxis == N_AXIS) {
                copyOutParams.blockCount = ubRealFactor_ * outH * outW;
            }
        }
        DataCopyPad(output_[outAddr], src[0], copyOutParams);
    }

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }
};
}

#endif // _IM2COL_NORMAL_NHWC_H_