/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_TO_SPACE_H
#define BATCH_TO_SPACE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "batch_to_space_tiling_data.h"

namespace NsBatchToSpace {
using namespace AscendC;
using namespace Ops::Base;

constexpr int32_t AXIS_N = 0;
constexpr int32_t AXIS_H = 1;
constexpr int32_t AXIS_W = 2;
constexpr int32_t AXIS_C = 3;
constexpr int32_t AXIS_COUNT = 4;
constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

// ======================================================================
// Layout 三区模型 (common_conversion.md §4.2)
// ======================================================================
struct AxisLayout {
    int64_t inStart{0};   // 动态设置：输入起始坐标
    int64_t outStart{0};  // 动态设置：输出起始坐标
    int64_t inStride{0};  // Init 固定：输入 stride（基于 inShape）
    int64_t outStride{0}; // Init 固定：输出 stride（基于 outShape）
};

struct Layout {
    AxisLayout axes[AXIS_COUNT];

    __aicore__ inline int64_t InOffset()
    {
        return axes[AXIS_N].inStart * axes[AXIS_N].inStride + axes[AXIS_H].inStart * axes[AXIS_H].inStride +
               axes[AXIS_W].inStart * axes[AXIS_W].inStride + axes[AXIS_C].inStart * axes[AXIS_C].inStride;
    }
    __aicore__ inline int64_t OutOffset()
    {
        return axes[AXIS_N].outStart * axes[AXIS_N].outStride + axes[AXIS_H].outStart * axes[AXIS_H].outStride +
               axes[AXIS_W].outStart * axes[AXIS_W].outStride + axes[AXIS_C].outStart * axes[AXIS_C].outStride;
    }
};

template <typename T, uint8_t UbAxis>
class BatchToSpace {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcOutputStart(int64_t idx, int64_t* coords);
    __aicore__ inline void CopyIn(int64_t idx, const int64_t* coords, int64_t bufOff);
    __aicore__ inline void CopyInAxisC(int64_t n, int64_t hOut, int64_t wOut, int64_t cStart, int64_t bufOff);
    __aicore__ inline void CopyInAxisW(int64_t n, int64_t hOut, int64_t wStart, int64_t wCount, int64_t bufOff);
    __aicore__ inline void CopyInAxisH(int64_t n, int64_t hStart, int64_t hCount, int64_t bufOff);
    __aicore__ inline void CopyInAxisN(int64_t nStart, int64_t nCount, int64_t bufOff);
    __aicore__ inline void InsertSync(const HardEvent& event);

    // 将 virtual 6D 坐标映射为真实 4D inStart，写入 layout_
    __aicore__ inline void SetInStart(int64_t n, int64_t bh, int64_t bw, int64_t hIn, int64_t wIn, int64_t c)
    {
        int64_t N = td_->outShape[AXIS_N];
        layout_.axes[AXIS_N].inStart = bh * bs_ * N + bw * N + n;
        layout_.axes[AXIS_H].inStart = hIn;
        layout_.axes[AXIS_W].inStart = wIn;
        layout_.axes[AXIS_C].inStart = c;
    }
    __aicore__ inline void SetOutStart(int64_t n, int64_t h, int64_t w, int64_t c)
    {
        layout_.axes[AXIS_N].outStart = n;
        layout_.axes[AXIS_H].outStart = h;
        layout_.axes[AXIS_W].outStart = w;
        layout_.axes[AXIS_C].outStart = c;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuffer_;
    GlobalTensor<T> inputGM_, outputGM_;
    const BatchToSpaceTilingData* td_{nullptr};

    Layout layout_;

    int64_t pingpongOffset_{0};
    int64_t blockIdx_{0};

    // crop / shape / alignment（坐标映射核心参数）
    int64_t bs_{0};
    int64_t ct_{0}, cb_{0}, cl_{0}, cr_{0};
    int64_t HIn_{0}, WIn_{0}, C_{0};
    int64_t HOut_{0}, WOut_{0};
    int64_t cAlignedBytes_{0};
    int64_t cAligned_{0};

    // virtual bw stride — DataCopyPad 的 srcStride/loop1SrcStride 参数，不属于 4D Layout
    int64_t Sbw_{0};
    int64_t N_{0};
};

template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::Init(GM_ADDR x, GM_ADDR y, const BatchToSpaceTilingData* tilingData)
{
    td_ = tilingData;
    inputGM_.SetGlobalBuffer((__gm__ T*)x);
    outputGM_.SetGlobalBuffer((__gm__ T*)y);
    blockIdx_ = GetBlockIdx();

    // Cache shape and crop values
    bs_ = td_->blockSize;
    ct_ = td_->cropTop;
    cb_ = td_->cropBottom;
    cl_ = td_->cropLeft;
    cr_ = td_->cropRight;
    HIn_ = td_->inShape[AXIS_H];
    WIn_ = td_->inShape[AXIS_W];
    C_ = td_->outShape[AXIS_C];
    HOut_ = td_->outShape[AXIS_H];
    WOut_ = td_->outShape[AXIS_W];

    cAlignedBytes_ = CeilAlign(C_ * static_cast<int64_t>(sizeof(T)), UB_BLOCK_BYTES);
    cAligned_ = cAlignedBytes_ / sizeof(T);

    // ── Layout: 真实 4D 输入 strides ──
    layout_.axes[AXIS_N].inStride = HIn_ * WIn_ * C_;
    layout_.axes[AXIS_H].inStride = WIn_ * C_;
    layout_.axes[AXIS_W].inStride = C_;
    layout_.axes[AXIS_C].inStride = 1;

    // ── Layout: 真实 4D 输出 strides ──
    layout_.axes[AXIS_N].outStride = HOut_ * WOut_ * C_;
    layout_.axes[AXIS_H].outStride = WOut_ * C_;
    layout_.axes[AXIS_W].outStride = C_;
    layout_.axes[AXIS_C].outStride = 1;

    // virtual bw stride (唯一不属于 4D Layout 的 stride)
    N_ = td_->outShape[AXIS_N];
    Sbw_ = N_ * HIn_ * WIn_ * C_;

    pingpongOffset_ = td_->bufferSize / sizeof(T);
    pipe_.InitBuffer(ubBuffer_, td_->bufferSize * BUFFER_NUM);
}

template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::InsertSync(const HardEvent& event)
{
    event_t eventID = static_cast<event_t>(pipe_.FetchEventID(event));
    switch (event) {
        case HardEvent::MTE2_MTE3:
            SetFlag<HardEvent::MTE2_MTE3>(eventID);
            WaitFlag<HardEvent::MTE2_MTE3>(eventID);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(eventID);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID);
            break;
        default:
            break;
    }
}

template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::CalcOutputStart(int64_t idx, int64_t* coords)
{
    coords[AXIS_C] = 0;

    if constexpr (UbAxis == AXIS_C) {
        int64_t cBlocks = CeilDiv(C_, static_cast<int64_t>(td_->ubFactor));
        int64_t group = idx / cBlocks;
        coords[AXIS_N] = group / (HOut_ * WOut_);
        int64_t rem = group % (HOut_ * WOut_);
        coords[AXIS_H] = rem / WOut_;
        coords[AXIS_W] = rem % WOut_;
    } else if constexpr (UbAxis == AXIS_W) {
        int64_t wBlocks = CeilDiv(WOut_, static_cast<int64_t>(td_->ubFactor));
        coords[AXIS_N] = idx / (HOut_ * wBlocks);
        int64_t rem = idx % (HOut_ * wBlocks);
        coords[AXIS_H] = rem / wBlocks;
        coords[AXIS_W] = (rem % wBlocks) * td_->ubFactor;
    } else if constexpr (UbAxis == AXIS_H) {
        int64_t hBlocks = CeilDiv(HOut_, static_cast<int64_t>(td_->ubFactor));
        coords[AXIS_N] = idx / hBlocks;
        coords[AXIS_H] = (idx % hBlocks) * td_->ubFactor;
        coords[AXIS_W] = 0;
    } else {
        coords[AXIS_N] = idx * td_->ubFactor;
        coords[AXIS_H] = 0;
        coords[AXIS_W] = 0;
    }
}

// ======================================================================
// CopyIn dispatch by ubAxis
// ======================================================================
template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::CopyIn(int64_t idx, const int64_t* coords, int64_t bufOff)
{
    if constexpr (UbAxis == AXIS_C) {
        int64_t cBlocks = CeilDiv(C_, static_cast<int64_t>(td_->ubFactor));
        int64_t cStart = (idx % cBlocks) * td_->ubFactor;
        CopyInAxisC(coords[AXIS_N], coords[AXIS_H], coords[AXIS_W], cStart, bufOff);
    } else if constexpr (UbAxis == AXIS_W) {
        int64_t wCount = WOut_ - coords[AXIS_W];
        wCount = (wCount > td_->ubFactor) ? td_->ubFactor : wCount;
        CopyInAxisW(coords[AXIS_N], coords[AXIS_H], coords[AXIS_W], wCount, bufOff);
    } else if constexpr (UbAxis == AXIS_H) {
        int64_t hCount = HOut_ - coords[AXIS_H];
        hCount = (hCount > td_->ubFactor) ? td_->ubFactor : hCount;
        CopyInAxisH(coords[AXIS_N], coords[AXIS_H], hCount, bufOff);
    } else {
        int64_t totalN = td_->outShape[AXIS_N];
        int64_t nCount = totalN - coords[AXIS_N];
        nCount = (nCount > td_->ubFactor) ? td_->ubFactor : nCount;
        CopyInAxisN(coords[AXIS_N], nCount, bufOff);
    }
}

// ======================================================================
// CopyInAxisC - ubAxis = 3: simple contiguous copy
// ======================================================================
template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::CopyInAxisC(int64_t n, int64_t hOut, int64_t wOut, int64_t cStart,
                                                            int64_t bufOff)
{
    int64_t cCount = C_ - cStart;
    cCount = (cCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : cCount;

    int64_t hFull = hOut + ct_;
    int64_t wFull = wOut + cl_;
    int64_t hIn = hFull / bs_;
    int64_t bh = hFull % bs_;
    int64_t wIn = wFull / bs_;
    int64_t bw = wFull % bs_;

    SetInStart(n, bh, bw, hIn, wIn, cStart);
    int64_t inOff = layout_.InOffset();

    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
    DataCopyExtParams copyParams;
    copyParams.blockLen = cCount * sizeof(T);
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(ubLocal[bufOff], inputGM_[inOff], copyParams, padParams);
}

// ======================================================================
// CopyInAxisW - ubAxis = 2: most common case
// ======================================================================
template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::CopyInAxisW(int64_t n, int64_t hOut, int64_t wStart, int64_t wCount,
                                                            int64_t bufOff)
{
    int64_t hFull = hOut + ct_;
    int64_t hIn = hFull / bs_;
    int64_t bh = hFull % bs_;

    int64_t wFullFirst = wStart + cl_;
    int64_t wInFirst = wFullFirst / bs_;
    int64_t bwFirst = wFullFirst % bs_;

    int64_t wFullLast = wStart + wCount - 1 + cl_;
    int64_t wInLast = wFullLast / bs_;
    int64_t bwLast = wFullLast % bs_;

    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
    int64_t cBytes = C_ * sizeof(T);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if (wInFirst == wInLast) {
        int64_t bwCount = bwLast - bwFirst + 1;
        SetInStart(n, bh, bwFirst, hIn, wInFirst, 0);
        int64_t inOff = layout_.InOffset();

        DataCopyExtParams copyParams;
        copyParams.blockLen = cBytes;
        copyParams.blockCount = bwCount;
        copyParams.srcStride = Sbw_ * sizeof(T) - cBytes;
        copyParams.dstStride = (cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;
        DataCopyPad(ubLocal[bufOff], inputGM_[inOff], copyParams, padParams);
    } else {
        int64_t curUbOff = bufOff;

        if (bwFirst > 0) {
            int64_t bwCount = bs_ - bwFirst;
            SetInStart(n, bh, bwFirst, hIn, wInFirst, 0);
            int64_t inOff = layout_.InOffset();

            DataCopyExtParams copyParams;
            copyParams.blockLen = cBytes;
            copyParams.blockCount = bwCount;
            copyParams.srcStride = Sbw_ * sizeof(T) - cBytes;
            copyParams.dstStride = (cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;
            DataCopyPad(ubLocal[curUbOff], inputGM_[inOff], copyParams, padParams);
            curUbOff += bwCount * cAligned_;
        }

        int64_t wInMiddleStart = (bwFirst > 0) ? wInFirst + 1 : wInFirst;
        int64_t numMiddleW = 0;
        if (bwLast < bs_ - 1) {
            if (wInLast > wInMiddleStart) {
                numMiddleW = wInLast - wInMiddleStart;
            }
        } else {
            numMiddleW = wInLast - wInMiddleStart + 1;
        }

        if (numMiddleW > 0) {
            SetInStart(n, bh, 0, hIn, wInMiddleStart, 0);
            int64_t inOff = layout_.InOffset();

            DataCopyExtParams copyParams;
            copyParams.blockLen = cBytes;
            copyParams.blockCount = numMiddleW;
            copyParams.srcStride = 0;
            copyParams.dstStride = (bs_ * cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;

            LoopModeParams loopParams;
            loopParams.loop1Size = bs_;
            loopParams.loop1SrcStride = Sbw_ * sizeof(T);
            loopParams.loop1DstStride = cAlignedBytes_;
            loopParams.loop2Size = 1;
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;

            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(ubLocal[curUbOff], inputGM_[inOff], copyParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            curUbOff += numMiddleW * bs_ * cAligned_;
        }

        if (bwLast < bs_ - 1) {
            int64_t bwCount = bwLast + 1;
            SetInStart(n, bh, 0, hIn, wInLast, 0);
            int64_t inOff = layout_.InOffset();

            DataCopyExtParams copyParams;
            copyParams.blockLen = cBytes;
            copyParams.blockCount = bwCount;
            copyParams.srcStride = Sbw_ * sizeof(T) - cBytes;
            copyParams.dstStride = (cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;
            DataCopyPad(ubLocal[curUbOff], inputGM_[inOff], copyParams, padParams);
        }
    }
}

// ======================================================================
// CopyInAxisH - ubAxis = 1: delegate each row to CopyInAxisW
// ======================================================================
template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::CopyInAxisH(int64_t n, int64_t hStart, int64_t hCount, int64_t bufOff)
{
    int64_t curUbOff = bufOff;
    for (int64_t hOut = hStart; hOut < hStart + hCount; ++hOut) {
        CopyInAxisW(n, hOut, 0, WOut_, curUbOff);
        curUbOff += WOut_ * cAligned_;
    }
}

// ======================================================================
// CopyInAxisN - ubAxis = 0: delegate each frame to CopyInAxisH
// ======================================================================
template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::CopyInAxisN(int64_t nStart, int64_t nCount, int64_t bufOff)
{
    int64_t frameSize = HOut_ * WOut_ * cAligned_;
    int64_t curUbOff = bufOff;
    for (int64_t n = nStart; n < nStart + nCount; ++n) {
        CopyInAxisH(n, 0, HOut_, curUbOff);
        curUbOff += frameSize;
    }
}

// ======================================================================
// Process - main loop with Ping-Pong
// ======================================================================
template <typename T, uint8_t UbAxis>
__aicore__ inline void BatchToSpace<T, UbAxis>::Process()
{
    for (int32_t i = 0; i < AXIS_COUNT; ++i) {
        if (td_->outShape[i] == 0)
            return;
    }

    int64_t beginIdx = blockIdx_ * td_->perCoreCount;
    int64_t endIdx = beginIdx + td_->perCoreCount;
    if (endIdx > static_cast<int64_t>(td_->totalCount))
        endIdx = td_->totalCount;
    if (beginIdx >= static_cast<int64_t>(td_->totalCount))
        return;

    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
    int64_t cBytes = C_ * sizeof(T);

    for (int64_t idx = beginIdx; idx < endIdx; ++idx) {
        int64_t bufOff = ((idx - beginIdx) % BUFFER_NUM) * pingpongOffset_;
        int64_t coords[AXIS_COUNT];

        CalcOutputStart(idx, coords);
        CopyIn(idx, coords, bufOff);
        InsertSync(HardEvent::MTE2_MTE3);

        // ── COPY_OUT: 统一通过 Layout 计算 outOffset ──
        if constexpr (UbAxis == AXIS_C) {
            int64_t cBlocks = CeilDiv(C_, static_cast<int64_t>(td_->ubFactor));
            int64_t cBlock = idx % cBlocks;
            int64_t cStart = cBlock * td_->ubFactor;
            int64_t cCount = C_ - cStart;
            cCount = (cCount > td_->ubFactor) ? td_->ubFactor : cCount;
            SetOutStart(coords[AXIS_N], coords[AXIS_H], coords[AXIS_W], cStart);
            DataCopyExtParams outParams;
            outParams.blockLen = cCount * sizeof(T);
            outParams.blockCount = 1;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(outputGM_[layout_.OutOffset()], ubLocal[bufOff], outParams);
        } else if constexpr (UbAxis == AXIS_W) {
            int64_t wCount = WOut_ - coords[AXIS_W];
            wCount = (wCount > td_->ubFactor) ? td_->ubFactor : wCount;
            SetOutStart(coords[AXIS_N], coords[AXIS_H], coords[AXIS_W], 0);
            DataCopyExtParams outParams;
            outParams.blockLen = cBytes;
            outParams.blockCount = wCount;
            outParams.srcStride = (cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;
            outParams.dstStride = 0;
            DataCopyPad(outputGM_[layout_.OutOffset()], ubLocal[bufOff], outParams);
        } else if constexpr (UbAxis == AXIS_H) {
            int64_t hCount = HOut_ - coords[AXIS_H];
            hCount = (hCount > td_->ubFactor) ? td_->ubFactor : hCount;
            SetOutStart(coords[AXIS_N], coords[AXIS_H], 0, 0);
            DataCopyExtParams outParams;
            outParams.blockLen = cBytes;
            outParams.blockCount = hCount * WOut_;
            outParams.srcStride = (cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;
            outParams.dstStride = 0;
            DataCopyPad(outputGM_[layout_.OutOffset()], ubLocal[bufOff], outParams);
        } else {
            int64_t nCount = td_->outShape[AXIS_N] - coords[AXIS_N];
            nCount = (nCount > td_->ubFactor) ? td_->ubFactor : nCount;
            SetOutStart(coords[AXIS_N], 0, 0, 0);
            DataCopyExtParams outParams;
            outParams.blockLen = cBytes;
            outParams.blockCount = nCount * HOut_ * WOut_;
            outParams.srcStride = (cAlignedBytes_ - cBytes) / UB_BLOCK_BYTES;
            outParams.dstStride = 0;
            DataCopyPad(outputGM_[layout_.OutOffset()], ubLocal[bufOff], outParams);
        }
        InsertSync(HardEvent::MTE3_MTE2);
    }
}

} // namespace NsBatchToSpace

#endif // BATCH_TO_SPACE_H
