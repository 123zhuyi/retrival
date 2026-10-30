/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef SLICE_WITH_AXES_H
#define SLICE_WITH_AXES_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "slice_with_axes_tiling_data.h"

namespace NsSliceWithAxes {
using namespace AscendC;
using namespace Ops::Base;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

template <typename T, uint8_t UbAxis>
class KernelSliceWithAxes {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SliceWithAxesTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoCopyIn(int64_t bufOff, int64_t axisLen, int64_t gmInOff);
    __aicore__ inline void DoCopyOut(int64_t bufOff, int64_t axisLen, int64_t gmOutOff);
    __aicore__ inline void InsertSync(const HardEvent& event);

    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    const SliceWithAxesTilingData* td_{nullptr};
    int32_t rank_{0};
    int64_t alignedW_{0};
    int64_t pingpongOff_{0};
    int64_t inStride_[MAX_AXIS_COUNT];
    int64_t outStride_[MAX_AXIS_COUNT];
};

template <typename T, uint8_t UbAxis>
__aicore__ inline void KernelSliceWithAxes<T, UbAxis>::Init(
    GM_ADDR x, GM_ADDR y, const SliceWithAxesTilingData* tilingData)
{
    td_ = tilingData;
    rank_ = static_cast<int32_t>(td_->rank);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    int32_t lastAxis = rank_ - 1;
    alignedW_ = CeilAlign(td_->outShape[lastAxis] * static_cast<int64_t>(sizeof(T)), UB_BLOCK_BYTES);
    pingpongOff_ = td_->bufferSize / sizeof(T);
    pipe_.InitBuffer(ubBuffer_, td_->bufferSize * BUFFER_NUM);

    inStride_[lastAxis] = 1;
    outStride_[lastAxis] = 1;
    for (int32_t ax = lastAxis - 1; ax >= 0; --ax) {
        inStride_[ax] = inStride_[ax + 1] * td_->inShape[ax + 1];
        outStride_[ax] = outStride_[ax + 1] * td_->outShape[ax + 1];
    }
}

template <typename T, uint8_t UbAxis>
__aicore__ inline void KernelSliceWithAxes<T, UbAxis>::InsertSync(const HardEvent& event)
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
__aicore__ inline void KernelSliceWithAxes<T, UbAxis>::DoCopyIn(int64_t bufOff, int64_t axisLen, int64_t gmInOff)
{
    int32_t lastAxis = rank_ - 1;
    int32_t numInner = rank_ - static_cast<int32_t>(UbAxis);

    int64_t blockLen = td_->outShape[lastAxis] * static_cast<int64_t>(sizeof(T));
    if (numInner == 1) {
        blockLen = axisLen * static_cast<int64_t>(sizeof(T));
    }

    int64_t blockCount = 1;
    int64_t srcStrideGap = 0;
    int64_t dstStrideGap = 0;
    if (numInner >= 2) {
        int32_t bcAxis = lastAxis - 1;
        blockCount = (bcAxis == static_cast<int32_t>(UbAxis)) ? axisLen : td_->outShape[bcAxis];
        srcStrideGap = (td_->inShape[lastAxis] - td_->outShape[lastAxis]) * static_cast<int64_t>(sizeof(T));
        dstStrideGap = (alignedW_ - blockLen) / UB_BLOCK_BYTES;
    }

    int64_t loop1Size = 1;
    int64_t loop1SrcStride = 0;
    int64_t loop1DstStride = 0;
    if (numInner >= 3) {
        int32_t l1Axis = lastAxis - 2;
        loop1Size = (l1Axis == static_cast<int32_t>(UbAxis)) ? axisLen : td_->outShape[l1Axis];
        loop1SrcStride = inStride_[l1Axis] * static_cast<int64_t>(sizeof(T));
        loop1DstStride = blockCount * alignedW_;
    }

    int64_t loop2Size = 1;
    int64_t loop2SrcStride = 0;
    int64_t loop2DstStride = 0;
    if (numInner >= 4) {
        int32_t l2Axis = lastAxis - 3;
        loop2Size = (l2Axis == static_cast<int32_t>(UbAxis)) ? axisLen : td_->outShape[l2Axis];
        loop2SrcStride = inStride_[l2Axis] * static_cast<int64_t>(sizeof(T));
        loop2DstStride = loop1Size * blockCount * alignedW_;
    }

    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(blockCount);
    copyParams.blockLen = static_cast<uint32_t>(blockLen);
    copyParams.srcStride = static_cast<uint32_t>(srcStrideGap);
    copyParams.dstStride = static_cast<uint32_t>(dstStrideGap);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if (numInner <= 2) {
        DataCopyPad(ubLocal[bufOff], xGm_[gmInOff], copyParams, padParams);
    } else if (numInner <= 4) {
        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(loop1Size);
        loopParams.loop2Size = static_cast<uint32_t>(loop2Size);
        loopParams.loop1SrcStride = static_cast<uint64_t>(loop1SrcStride);
        loopParams.loop2SrcStride = static_cast<uint64_t>(loop2SrcStride);
        loopParams.loop1DstStride = static_cast<uint64_t>(loop1DstStride);
        loopParams.loop2DstStride = static_cast<uint64_t>(loop2DstStride);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(ubLocal[bufOff], xGm_[gmInOff], copyParams, padParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {
        int64_t innerUbChunk = loop2Size * loop1Size * blockCount * (alignedW_ / static_cast<int64_t>(sizeof(T)));

        int32_t forStart = static_cast<int32_t>(UbAxis);
        int32_t forEnd = lastAxis - 4;

        int64_t forTotal = 1;
        for (int32_t ax = forStart; ax <= forEnd; ++ax) {
            forTotal *= (ax == static_cast<int32_t>(UbAxis)) ? axisLen : td_->outShape[ax];
        }

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(loop1Size);
        loopParams.loop2Size = static_cast<uint32_t>(loop2Size);
        loopParams.loop1SrcStride = static_cast<uint64_t>(loop1SrcStride);
        loopParams.loop2SrcStride = static_cast<uint64_t>(loop2SrcStride);
        loopParams.loop1DstStride = static_cast<uint64_t>(loop1DstStride);
        loopParams.loop2DstStride = static_cast<uint64_t>(loop2DstStride);

        for (int64_t fi = 0; fi < forTotal; ++fi) {
            int64_t forGmOff = 0;
            int64_t rem = fi;
            for (int32_t ax = forEnd; ax >= forStart; --ax) {
                int64_t dimSize = (ax == static_cast<int32_t>(UbAxis)) ? axisLen : td_->outShape[ax];
                int64_t coord = rem % dimSize;
                rem /= dimSize;
                forGmOff += coord * inStride_[ax];
            }

            int64_t ubOff = bufOff + fi * innerUbChunk;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(ubLocal[ubOff], xGm_[gmInOff + forGmOff], copyParams, padParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
    }
}

template <typename T, uint8_t UbAxis>
__aicore__ inline void KernelSliceWithAxes<T, UbAxis>::DoCopyOut(int64_t bufOff, int64_t axisLen, int64_t gmOutOff)
{
    int32_t lastAxis = rank_ - 1;
    int32_t numInner = rank_ - static_cast<int32_t>(UbAxis);

    int64_t blockLen;
    int64_t totalRows;

    if (numInner == 1) {
        blockLen = axisLen * static_cast<int64_t>(sizeof(T));
        totalRows = 1;
    } else {
        blockLen = td_->outShape[lastAxis] * static_cast<int64_t>(sizeof(T));
        totalRows = axisLen;
        for (int32_t ax = static_cast<int32_t>(UbAxis) + 1; ax < lastAxis; ++ax) {
            totalRows *= td_->outShape[ax];
        }
    }

    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
    DataCopyExtParams outParams;
    outParams.blockCount = static_cast<uint16_t>(totalRows);
    outParams.blockLen = static_cast<uint32_t>(blockLen);
    outParams.srcStride = (numInner <= 1) ? 0 : static_cast<uint32_t>((alignedW_ - blockLen) / UB_BLOCK_BYTES);
    outParams.dstStride = 0;
    DataCopyPad(yGm_[gmOutOff], ubLocal[bufOff], outParams);
}

template <typename T, uint8_t UbAxis>
__aicore__ inline void KernelSliceWithAxes<T, UbAxis>::Process()
{
    for (int32_t i = 0; i < rank_; ++i) {
        if (td_->outShape[i] == 0)
            return;
    }

    int64_t beginIdx = GetBlockIdx() * td_->perCoreCount;
    int64_t endIdx = beginIdx + td_->perCoreCount;
    if (endIdx > static_cast<int64_t>(td_->totalCount)) {
        endIdx = td_->totalCount;
    }
    if (beginIdx >= static_cast<int64_t>(td_->totalCount)) {
        return;
    }

    int64_t ubBlocks = CeilDiv(static_cast<int64_t>(td_->outShape[UbAxis]), static_cast<int64_t>(td_->ubFactor));

    for (int64_t idx = beginIdx; idx < endIdx; ++idx) {
        int64_t bufOff = ((idx - beginIdx) % BUFFER_NUM) * pingpongOff_;

        int64_t ubBlockIdx = idx % ubBlocks;
        int64_t outerIdx = idx / ubBlocks;

        int64_t coords[MAX_AXIS_COUNT] = {0};
        for (int32_t ax = static_cast<int32_t>(UbAxis) - 1; ax >= 0; --ax) {
            coords[ax] = outerIdx % td_->outShape[ax];
            outerIdx /= td_->outShape[ax];
        }

        int64_t ubStart = ubBlockIdx * td_->ubFactor;
        int64_t axisLen = td_->outShape[UbAxis] - ubStart;
        if (axisLen > static_cast<int64_t>(td_->ubFactor)) {
            axisLen = td_->ubFactor;
        }
        coords[UbAxis] = ubStart;

        int64_t gmInOff = 0;
        int64_t gmOutOff = 0;
        for (int32_t ax = 0; ax <= static_cast<int32_t>(UbAxis); ++ax) {
            gmInOff += (coords[ax] + td_->fullOffsets[ax]) * inStride_[ax];
            gmOutOff += coords[ax] * outStride_[ax];
        }
        for (int32_t ax = static_cast<int32_t>(UbAxis) + 1; ax < rank_; ++ax) {
            gmInOff += td_->fullOffsets[ax] * inStride_[ax];
        }

        DoCopyIn(bufOff, axisLen, gmInOff);
        InsertSync(HardEvent::MTE2_MTE3);
        DoCopyOut(bufOff, axisLen, gmOutOff);
        InsertSync(HardEvent::MTE3_MTE2);
    }
}

} // namespace NsSliceWithAxes

#endif // SLICE_WITH_AXES_H
