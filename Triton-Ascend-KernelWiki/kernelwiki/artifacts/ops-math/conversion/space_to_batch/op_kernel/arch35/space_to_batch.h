/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPACE_TO_BATCH_H
#define SPACE_TO_BATCH_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "space_to_batch_tiling_data.h"

namespace STB {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, int UbAxis = STB_AXIS_C>
class SpaceToBatchKernel {
public:
    // ========================================================================
    //  Init
    // ========================================================================

    __aicore__ inline SpaceToBatchKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SpaceToBatchTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();

        inputGm_.SetGlobalBuffer((__gm__ T*)x);
        outputGm_.SetGlobalBuffer((__gm__ T*)y);

        layout_.axes[STB_AXIS_C].inStride = 1;
        layout_.axes[STB_AXIS_W].inStride = tilingData_->inShape[STB_AXIS_C];
        layout_.axes[STB_AXIS_H].inStride = tilingData_->inShape[STB_AXIS_W] * layout_.axes[STB_AXIS_W].inStride;
        layout_.axes[STB_AXIS_N].inStride = tilingData_->inShape[STB_AXIS_H] * layout_.axes[STB_AXIS_H].inStride;

        layout_.axes[STB_AXIS_C].outStride = 1;
        layout_.axes[STB_AXIS_W].outStride = tilingData_->outShape[STB_AXIS_C];
        layout_.axes[STB_AXIS_H].outStride = tilingData_->outShape[STB_AXIS_W] * layout_.axes[STB_AXIS_W].outStride;
        layout_.axes[STB_AXIS_N].outStride = tilingData_->outShape[STB_AXIS_H] * layout_.axes[STB_AXIS_H].outStride;

        channelAligned_ = CeilAlign(static_cast<uint32_t>(tilingData_->inShape[STB_AXIS_C]),
                                    static_cast<uint32_t>(ELEMS_PER_UB_BLOCK));
        channelAlignedBytes_ = channelAligned_ * sizeof(T);

        pipe_->InitBuffer(ubBuffer_, tilingData_->bufferSize * BUFFER_NUM);
    }

    // ========================================================================
    //  Process (TBuf + Ping-Pong)
    // ========================================================================

    __aicore__ inline void Process()
    {
        if (tilingData_->totalCount == 0) {
            return;
        }

        uint32_t startIdx = blockIdx_ * tilingData_->perCoreCount;
        if (startIdx >= tilingData_->totalCount) {
            return;
        }

        uint32_t endIdx = (blockIdx_ + 1) * tilingData_->perCoreCount;
        endIdx = endIdx < tilingData_->totalCount ? endIdx : tilingData_->totalCount;

        LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
        for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
            uint32_t bufIdx = (idx - startIdx) & (BUFFER_NUM - 1);
            uint64_t ubOff = bufIdx * (tilingData_->bufferSize / sizeof(T));

            if (BlockHasPadding(idx)) {
                Duplicate(ubLocal[ubOff], static_cast<T>(0), tilingData_->bufferSize / sizeof(T));
            }
            SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DoCopyIn(ubLocal, ubOff, idx);
            SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
            DoCopyOut(ubLocal, ubOff, idx);
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }

private:
    // ========================================================================
    //  Layout (three-zone model)
    // ========================================================================

    struct AxisLayout {
        int64_t inStart;
        int64_t outStart;
        int64_t length;
        int64_t inStride;
        int64_t outStride;
    };

    struct Layout {
        AxisLayout axes[STB_AXIS_COUNT];
    } layout_;

    // ========================================================================
    //  Coordinate mapping
    // ========================================================================

    __aicore__ inline void ComputeOutIndex(uint32_t blockIdx, uint64_t* outIndex) const
    {
        uint64_t linearIdx = blockIdx;
        for (int32_t axis = UbAxis; axis >= 0; --axis) {
            uint64_t dim = tilingData_->outShape[axis];
            if (axis == UbAxis) {
                dim = CeilDiv(dim, static_cast<uint64_t>(tilingData_->ubFactor));
            }
            if (dim > 0) {
                outIndex[axis] = (axis == UbAxis) ? (linearIdx % dim) * tilingData_->ubFactor : linearIdx % dim;
                linearIdx /= dim;
            }
        }
    }

    __aicore__ inline void DecodeNOut(int64_t nOut, int64_t& n, int64_t& bh, int64_t& bw) const
    {
        int64_t blockSize = tilingData_->blockSize;
        int64_t N = tilingData_->inShape[STB_AXIS_N];
        n = nOut % N;
        int64_t rem = nOut / N;
        bw = rem % blockSize;
        bh = rem / blockSize;
    }

    __aicore__ inline bool MapToInput(int64_t nOut, int64_t hOut, int64_t wOut, int64_t& n, int64_t& hIn,
                                      int64_t& wIn) const
    {
        int64_t bh, bw;
        DecodeNOut(nOut, n, bh, bw);
        int64_t blockSize = tilingData_->blockSize;
        hIn = hOut * blockSize + bh - tilingData_->paddings[0][0];
        wIn = wOut * blockSize + bw - tilingData_->paddings[1][0];
        return (hIn >= 0 && hIn < tilingData_->inShape[STB_AXIS_H] && wIn >= 0 &&
                wIn < tilingData_->inShape[STB_AXIS_W]);
    }

    // ========================================================================
    //  BlockHasPadding — skip Duplicate when block has no padding edges
    // ========================================================================

    __aicore__ inline bool BlockHasPadding(uint32_t blockIdx) const
    {
        uint64_t outIdx[STB_AXIS_COUNT] = {};
        ComputeOutIndex(blockIdx, outIdx);
        int64_t nOut = static_cast<int64_t>(outIdx[STB_AXIS_N]);
        int64_t hOut = static_cast<int64_t>(outIdx[STB_AXIS_H]);
        int64_t wStart = static_cast<int64_t>(outIdx[STB_AXIS_W]);

        if constexpr (UbAxis == STB_AXIS_W) {
            int64_t n, bh, bw;
            DecodeNOut(nOut, n, bh, bw);
            int64_t blockSize = tilingData_->blockSize;
            int64_t hIn = hOut * blockSize + bh - tilingData_->paddings[0][0];
            if (hIn < 0 || hIn >= tilingData_->inShape[STB_AXIS_H]) {
                return true;
            }
            int64_t padLeft = tilingData_->paddings[1][0];
            int64_t widthIn = tilingData_->inShape[STB_AXIS_W];
            int64_t wValidMin = (padLeft - bw + blockSize - 1) / blockSize;
            if (wValidMin < 0) {
                wValidMin = 0;
            }
            int64_t wValidMax = (widthIn + padLeft - bw - 1) / blockSize;
            int64_t wEnd = wStart + tilingData_->ubFactor - 1;
            int64_t wOutLast = tilingData_->outShape[STB_AXIS_W] - 1;
            if (wEnd > wOutLast) {
                wEnd = wOutLast;
            }
            return (wStart < wValidMin || wEnd > wValidMax);
        }

        if constexpr (UbAxis == STB_AXIS_C) {
            int64_t n, hIn, wIn;
            return !MapToInput(nOut, hOut, wStart, n, hIn, wIn);
        }

        // H / N axis: conservatively always Duplicate
        return true;
    }

    // ========================================================================
    //  DoCopyIn / DoCopyOut
    // ========================================================================

    __aicore__ inline void DoCopyIn(LocalTensor<T>& ubLocal, uint64_t ubOff, uint32_t blockIdx)
    {
        uint64_t outIdx[STB_AXIS_COUNT] = {};
        ComputeOutIndex(blockIdx, outIdx);
        int64_t nOut = static_cast<int64_t>(outIdx[STB_AXIS_N]);
        int64_t hOut = static_cast<int64_t>(outIdx[STB_AXIS_H]);
        int64_t wStart = static_cast<int64_t>(outIdx[STB_AXIS_W]);
        int64_t cStart = static_cast<int64_t>(outIdx[STB_AXIS_C]);

        if constexpr (UbAxis == STB_AXIS_C) {
            CopyInAxisC(ubLocal, ubOff, nOut, hOut, wStart, cStart);
        } else if constexpr (UbAxis == STB_AXIS_W) {
            CopyInAxisW(ubLocal, ubOff, nOut, hOut, wStart, tilingData_->ubFactor);
        } else if constexpr (UbAxis == STB_AXIS_H) {
            CopyInAxisH(ubLocal, ubOff, nOut, hOut, tilingData_->ubFactor);
        } else {
            CopyInAxisN(ubLocal, ubOff, nOut, tilingData_->ubFactor);
        }
    }

    __aicore__ inline void DoCopyOut(LocalTensor<T>& ubLocal, uint64_t ubOff, uint32_t blockIdx)
    {
        uint64_t outIdx[STB_AXIS_COUNT] = {};
        ComputeOutIndex(blockIdx, outIdx);
        int64_t nOut = static_cast<int64_t>(outIdx[STB_AXIS_N]);
        int64_t hOut = static_cast<int64_t>(outIdx[STB_AXIS_H]);
        int64_t wStart = static_cast<int64_t>(outIdx[STB_AXIS_W]);
        int64_t cStart = static_cast<int64_t>(outIdx[STB_AXIS_C]);

        uint64_t outputOff = static_cast<uint64_t>(
            nOut * layout_.axes[STB_AXIS_N].outStride + hOut * layout_.axes[STB_AXIS_H].outStride +
            wStart * layout_.axes[STB_AXIS_W].outStride + cStart * layout_.axes[STB_AXIS_C].outStride);

        uint32_t channelCount = static_cast<uint32_t>(tilingData_->inShape[STB_AXIS_C]);
        uint32_t channelBytes = channelCount * sizeof(T);

        if constexpr (UbAxis == STB_AXIS_C) {
            uint32_t copyCount = tilingData_->ubFactor;
            int64_t channelRemain = static_cast<int64_t>(channelCount) - cStart;
            if (channelRemain < static_cast<int64_t>(copyCount)) {
                copyCount = static_cast<uint32_t>(channelRemain);
            }
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = copyCount * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(outputGm_[outputOff], ubLocal[ubOff], copyParams);
        } else if constexpr (UbAxis == STB_AXIS_W) {
            uint32_t actualCount = tilingData_->ubFactor;
            int64_t remaining = tilingData_->outShape[STB_AXIS_W] - wStart;
            if (remaining < static_cast<int64_t>(actualCount)) {
                actualCount = static_cast<uint32_t>(remaining);
            }
            DataCopyExtParams copyParams;
            copyParams.blockLen = channelBytes;
            copyParams.blockCount = static_cast<uint16_t>(actualCount);
            copyParams.srcStride = (channelAlignedBytes_ - channelBytes) / UB_BLOCK_BYTES;
            copyParams.dstStride = 0;
            DataCopyPad(outputGm_[outputOff], ubLocal[ubOff], copyParams);
        } else if constexpr (UbAxis == STB_AXIS_H) {
            uint32_t hCount = tilingData_->ubFactor;
            int64_t hRemain = static_cast<int64_t>(tilingData_->outShape[STB_AXIS_H]) - hOut;
            if (hRemain < static_cast<int64_t>(hCount)) {
                hCount = static_cast<uint32_t>(hRemain);
            }
            uint32_t widthOutCount = static_cast<uint32_t>(tilingData_->outShape[STB_AXIS_W]);
            DataCopyExtParams copyParams;
            copyParams.blockLen = channelBytes;
            copyParams.blockCount = static_cast<uint16_t>(hCount * widthOutCount);
            copyParams.srcStride = (channelAlignedBytes_ - channelBytes) / UB_BLOCK_BYTES;
            copyParams.dstStride = 0;
            DataCopyPad(outputGm_[outputOff], ubLocal[ubOff], copyParams);
        } else { // STB_AXIS_N
            uint32_t nCount = tilingData_->ubFactor;
            int64_t nRemain = static_cast<int64_t>(tilingData_->outShape[STB_AXIS_N]) - nOut;
            if (nRemain < static_cast<int64_t>(nCount)) {
                nCount = static_cast<uint32_t>(nRemain);
            }
            uint32_t heightOutCount = static_cast<uint32_t>(tilingData_->outShape[STB_AXIS_H]);
            uint32_t widthOutCount = static_cast<uint32_t>(tilingData_->outShape[STB_AXIS_W]);
            DataCopyExtParams copyParams;
            copyParams.blockLen = channelBytes;
            copyParams.blockCount = static_cast<uint16_t>(nCount * heightOutCount * widthOutCount);
            copyParams.srcStride = (channelAlignedBytes_ - channelBytes) / UB_BLOCK_BYTES;
            copyParams.dstStride = 0;
            DataCopyPad(outputGm_[outputOff], ubLocal[ubOff], copyParams);
        }
    }

    // ========================================================================
    //  CopyIn — ubAxis = C
    // ========================================================================

    __aicore__ inline void CopyInAxisC(LocalTensor<T>& ubLocal, uint64_t ubOff, int64_t nOut, int64_t hOut,
                                       int64_t wOut, int64_t cStart)
    {
        int64_t n, hIn, wIn;
        if (!MapToInput(nOut, hOut, wOut, n, hIn, wIn)) {
            return; // UB already zeroed by Duplicate
        }
        int64_t channels = tilingData_->inShape[STB_AXIS_C];
        uint32_t copyCount = tilingData_->ubFactor;
        int64_t remaining = channels - cStart;
        if (remaining < static_cast<int64_t>(copyCount)) {
            copyCount = static_cast<uint32_t>(remaining);
        }
        uint64_t inputOff = static_cast<uint64_t>(
            n * layout_.axes[STB_AXIS_N].inStride + hIn * layout_.axes[STB_AXIS_H].inStride +
            wIn * layout_.axes[STB_AXIS_W].inStride + cStart * layout_.axes[STB_AXIS_C].inStride);

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = copyCount * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(ubLocal[ubOff], inputGm_[inputOff], copyParams, padParams);
    }

    // ========================================================================
    //  CopyIn — ubAxis = W
    //
    //  Single-split: each block has fixed n_out → fixed (n, bh, bw).
    //  LoopMode iterates validWCount positions, copying channels elements each,
    //  with source jumping blockSize*channels between w_out values.
    //  UB layout: each w_out gets an aligned row (channelAlignedBytes_ stride).
    // ========================================================================

    __aicore__ inline void CopyInAxisW(LocalTensor<T>& ubLocal, uint64_t ubOff, int64_t nOut, int64_t hOut,
                                       int64_t wStart, uint32_t count)
    {
        int64_t n, bh, bw;
        DecodeNOut(nOut, n, bh, bw);
        int64_t blockSize = tilingData_->blockSize;
        int64_t hIn = hOut * blockSize + bh - tilingData_->paddings[0][0];
        if (hIn < 0 || hIn >= tilingData_->inShape[STB_AXIS_H]) {
            return;
        }
        int64_t channels = tilingData_->inShape[STB_AXIS_C];
        int64_t widthIn = tilingData_->inShape[STB_AXIS_W];
        int64_t padLeft = tilingData_->paddings[1][0];

        int64_t wValidMin = (padLeft - bw + blockSize - 1) / blockSize;
        if (wValidMin < 0) {
            wValidMin = 0;
        }
        int64_t wValidMax = (widthIn + padLeft - bw - 1) / blockSize;

        int64_t wEnd = wStart + count - 1;
        int64_t wBlockStart = (wStart > wValidMin) ? wStart : wValidMin;
        int64_t wBlockEnd = (wEnd < wValidMax) ? wEnd : wValidMax;

        if (wBlockStart > wBlockEnd) {
            return;
        }
        int64_t validWCount = wBlockEnd - wBlockStart + 1;

        int64_t wInFirst = wBlockStart * blockSize + bw - padLeft;
        uint64_t inputAddr = static_cast<uint64_t>(n * layout_.axes[STB_AXIS_N].inStride +
                                                   hIn * layout_.axes[STB_AXIS_H].inStride +
                                                   wInFirst * layout_.axes[STB_AXIS_W].inStride);

        uint64_t curUbOff = ubOff + (wBlockStart - wStart) * channelAligned_;

        DataCopyExtParams copyParams;
        copyParams.blockLen = channels * sizeof(T);
        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(validWCount);
        loopParams.loop1SrcStride = static_cast<uint32_t>(blockSize * channels * sizeof(T));
        loopParams.loop1DstStride = channelAlignedBytes_;
        loopParams.loop2Size = 1;

        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(ubLocal[curUbOff], inputGm_[inputAddr], copyParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    // ========================================================================
    //  CopyIn — ubAxis = H
    // ========================================================================

    __aicore__ inline void CopyInAxisH(LocalTensor<T>& ubLocal, uint64_t ubOff, int64_t nOut, int64_t hStart,
                                       uint32_t count)
    {
        int64_t n, bh, bw;
        DecodeNOut(nOut, n, bh, bw);

        int64_t blockSize = tilingData_->blockSize;
        int64_t heightIn = tilingData_->inShape[STB_AXIS_H];
        int64_t padTop = tilingData_->paddings[0][0];
        int64_t heightOut = tilingData_->outShape[STB_AXIS_H];
        int64_t widthOut = tilingData_->outShape[STB_AXIS_W];

        int64_t hValidMin = (padTop - bh + blockSize - 1) / blockSize;
        if (hValidMin < 0) {
            hValidMin = 0;
        }
        int64_t hValidMax = (heightIn + padTop - bh - 1) / blockSize;
        if (hValidMax >= heightOut) {
            hValidMax = heightOut - 1;
        }

        int64_t hEnd = hStart + count - 1;
        int64_t hBlockStart = (hStart > hValidMin) ? hStart : hValidMin;
        int64_t hBlockEnd = (hEnd < hValidMax) ? hEnd : hValidMax;

        uint64_t curUbOff = ubOff;
        for (int64_t hOut = hStart; hOut < hBlockStart; ++hOut) {
            curUbOff += widthOut * channelAligned_;
        }
        for (int64_t hOut = hBlockStart; hOut <= hBlockEnd; ++hOut) {
            CopyInAxisW(ubLocal, curUbOff, nOut, hOut, 0, static_cast<uint32_t>(widthOut));
            curUbOff += widthOut * channelAligned_;
        }
    }

    // ========================================================================
    //  CopyIn — ubAxis = N
    // ========================================================================

    __aicore__ inline void CopyInAxisN(LocalTensor<T>& ubLocal, uint64_t ubOff, int64_t nStart, uint32_t count)
    {
        uint64_t frameSize = tilingData_->outShape[STB_AXIS_H] * tilingData_->outShape[STB_AXIS_W] * channelAligned_;
        for (uint32_t ni = 0; ni < count; ++ni) {
            int64_t nOut = nStart + ni;
            uint64_t frameUbOff = ubOff + ni * frameSize;
            CopyInAxisH(ubLocal, frameUbOff, nOut, 0, static_cast<uint32_t>(tilingData_->outShape[STB_AXIS_H]));
        }
    }

    // ========================================================================
    //  Synchronization
    // ========================================================================

    template <HardEvent EVENT>
    __aicore__ inline void SetEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<EVENT>(eventId);
        WaitFlag<EVENT>(eventId);
    }

    // ========================================================================
    //  Member variables
    // ========================================================================

    TPipe* pipe_ = nullptr;
    const SpaceToBatchTilingData* tilingData_ = nullptr;
    uint32_t blockIdx_{0};

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> ubBuffer_;

    uint32_t channelAligned_{0};
    uint32_t channelAlignedBytes_{0};

    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
    static constexpr uint32_t ELEMS_PER_UB_BLOCK = UB_BLOCK_BYTES / sizeof(T);
};

} // namespace STB

#endif // SPACE_TO_BATCH_H
