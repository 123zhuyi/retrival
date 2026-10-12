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

#ifndef SLICE_LAST_DIM_H
#define SLICE_LAST_DIM_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "slice_last_dim_tiling_data.h"

namespace NsSliceLastDim {
using namespace AscendC;
using namespace Ops::Base;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

template <typename T, uint8_t CopyMode, uint8_t UbAxis>
class SliceLastDim {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SliceLastDimTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoCopyIn(int64_t bufOff, int64_t idx);
    __aicore__ inline void DoCopyOut(int64_t bufOff, int64_t idx);
    __aicore__ inline void InsertSync(const HardEvent& event);

    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuffer_;
    GlobalTensor<T> xGm_, yGm_;
    const SliceLastDimTilingData* td_{nullptr};

    int64_t alignedW_{0};
    int64_t pingpongOff_{0};
};

template <typename T, uint8_t CopyMode, uint8_t UbAxis>
__aicore__ inline void SliceLastDim<T, CopyMode, UbAxis>::Init(
    GM_ADDR x, GM_ADDR y, const SliceLastDimTilingData* tilingData)
{
    td_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pingpongOff_ = td_->bufferSize / sizeof(T);
    pipe_.InitBuffer(ubBuffer_, td_->bufferSize * BUFFER_NUM);

    if constexpr (UbAxis == 0) {
        alignedW_ = CeilAlign(td_->lastDimOut * static_cast<int64_t>(sizeof(T)), UB_BLOCK_BYTES);
    } else {
        alignedW_ = 0;
    }
}

template <typename T, uint8_t CopyMode, uint8_t UbAxis>
__aicore__ inline void SliceLastDim<T, CopyMode, UbAxis>::InsertSync(const HardEvent& event)
{
    event_t id = static_cast<event_t>(pipe_.FetchEventID(event));
    switch (event) {
        case HardEvent::MTE2_MTE3:
            SetFlag<HardEvent::MTE2_MTE3>(id);
            WaitFlag<HardEvent::MTE2_MTE3>(id);
            break;
        case HardEvent::MTE3_MTE2:
            SetFlag<HardEvent::MTE3_MTE2>(id);
            WaitFlag<HardEvent::MTE3_MTE2>(id);
            break;
        default:
            break;
    }
}

template <typename T, uint8_t CopyMode, uint8_t UbAxis>
__aicore__ inline void SliceLastDim<T, CopyMode, UbAxis>::DoCopyIn(int64_t bufOff, int64_t idx)
{
    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    if constexpr (CopyMode == 0) {
        if constexpr (UbAxis == 0) {
            int64_t rowStart = idx * td_->ubFactor;
            int64_t rowCount = td_->outerSize - rowStart;
            rowCount = (rowCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : rowCount;
            int64_t inOff = rowStart * td_->lastDimIn + td_->start;
            int64_t blockLen = td_->lastDimOut * static_cast<int64_t>(sizeof(T));

            DataCopyExtParams copyParams;
            copyParams.blockLen = static_cast<uint32_t>(blockLen);
            copyParams.blockCount = static_cast<uint16_t>(rowCount);
            copyParams.srcStride =
                static_cast<uint32_t>((td_->lastDimIn - td_->lastDimOut) * static_cast<int64_t>(sizeof(T)));
            copyParams.dstStride = static_cast<uint32_t>((alignedW_ - blockLen) / UB_BLOCK_BYTES);
            DataCopyPad(ubLocal[bufOff], xGm_[inOff], copyParams, padParams);
        } else {
            int64_t outerIdx = idx / CeilDiv(td_->lastDimOut, static_cast<int64_t>(td_->ubFactor));
            int64_t sliceBlock = idx % CeilDiv(td_->lastDimOut, static_cast<int64_t>(td_->ubFactor));
            int64_t sliceStart = sliceBlock * td_->ubFactor;
            int64_t sliceCount = td_->lastDimOut - sliceStart;
            sliceCount = (sliceCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : sliceCount;
            int64_t inOff = outerIdx * td_->lastDimIn + td_->start + sliceStart;

            DataCopyExtParams copyParams;
            copyParams.blockLen = static_cast<uint32_t>(sliceCount * static_cast<int64_t>(sizeof(T)));
            copyParams.blockCount = 1;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(ubLocal[bufOff], xGm_[inOff], copyParams, padParams);
        }
    } else {
        if constexpr (UbAxis == 0) {
            int64_t rowStart = idx * td_->ubFactor;
            int64_t rowCount = td_->outerSize - rowStart;
            rowCount = (rowCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : rowCount;
            int64_t inOff = rowStart * td_->lastDimIn + td_->start;
            int64_t alignedOut = alignedW_ / static_cast<int64_t>(sizeof(T));

            NdDmaDci();
            NdDmaLoopInfo<2> loopInfo{};
            loopInfo.loopSrcStride[0] = static_cast<uint64_t>(td_->stride);
            loopInfo.loopDstStride[0] = 1;
            loopInfo.loopSize[0] = static_cast<uint32_t>(td_->lastDimOut);
            loopInfo.loopSrcStride[1] = static_cast<uint64_t>(td_->lastDimIn);
            loopInfo.loopDstStride[1] = static_cast<uint32_t>(alignedOut);
            loopInfo.loopSize[1] = static_cast<uint32_t>(rowCount);
            NdDmaParams<T, 2> params{loopInfo, 0};
            DataCopy<T, 2>(ubLocal[bufOff], xGm_[inOff], params);
        } else {
            int64_t outerIdx = idx / CeilDiv(td_->lastDimOut, static_cast<int64_t>(td_->ubFactor));
            int64_t sliceBlock = idx % CeilDiv(td_->lastDimOut, static_cast<int64_t>(td_->ubFactor));
            int64_t sliceStart = sliceBlock * td_->ubFactor;
            int64_t sliceCount = td_->lastDimOut - sliceStart;
            sliceCount = (sliceCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : sliceCount;
            int64_t inOff = outerIdx * td_->lastDimIn + td_->start + sliceStart * td_->stride;

            NdDmaDci();
            NdDmaLoopInfo<1> loopInfo{};
            loopInfo.loopSrcStride[0] = static_cast<uint64_t>(td_->stride);
            loopInfo.loopDstStride[0] = 1;
            loopInfo.loopSize[0] = static_cast<uint32_t>(sliceCount);
            NdDmaParams<T, 1> params{loopInfo, 0};
            DataCopy<T, 1>(ubLocal[bufOff], xGm_[inOff], params);
        }
    }
}

template <typename T, uint8_t CopyMode, uint8_t UbAxis>
__aicore__ inline void SliceLastDim<T, CopyMode, UbAxis>::DoCopyOut(int64_t bufOff, int64_t idx)
{
    LocalTensor<T> ubLocal = ubBuffer_.Get<T>();

    if constexpr (UbAxis == 0) {
        int64_t rowStart = idx * td_->ubFactor;
        int64_t rowCount = td_->outerSize - rowStart;
        rowCount = (rowCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : rowCount;
        int64_t outOff = rowStart * td_->lastDimOut;
        int64_t blockLen = td_->lastDimOut * static_cast<int64_t>(sizeof(T));

        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(blockLen);
        outParams.blockCount = static_cast<uint16_t>(rowCount);
        outParams.srcStride = static_cast<uint32_t>((alignedW_ - blockLen) / UB_BLOCK_BYTES);
        outParams.dstStride = 0;
        DataCopyPad(yGm_[outOff], ubLocal[bufOff], outParams);
    } else {
        int64_t outerIdx = idx / CeilDiv(td_->lastDimOut, static_cast<int64_t>(td_->ubFactor));
        int64_t sliceBlock = idx % CeilDiv(td_->lastDimOut, static_cast<int64_t>(td_->ubFactor));
        int64_t sliceStart = sliceBlock * td_->ubFactor;
        int64_t sliceCount = td_->lastDimOut - sliceStart;
        sliceCount = (sliceCount > static_cast<int64_t>(td_->ubFactor)) ? td_->ubFactor : sliceCount;
        int64_t outOff = outerIdx * td_->lastDimOut + sliceStart;

        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(sliceCount * static_cast<int64_t>(sizeof(T)));
        outParams.blockCount = 1;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        DataCopyPad(yGm_[outOff], ubLocal[bufOff], outParams);
    }
}

template <typename T, uint8_t CopyMode, uint8_t UbAxis>
__aicore__ inline void SliceLastDim<T, CopyMode, UbAxis>::Process()
{
    if (td_->outerSize == 0 || td_->lastDimOut == 0)
        return;

    uint64_t blockIdx = GetBlockIdx();
    uint64_t beginIdx = blockIdx * td_->perCoreCount;
    uint64_t endIdx = beginIdx + td_->perCoreCount;
    if (endIdx > td_->totalCount) {
        endIdx = td_->totalCount;
    }
    if (beginIdx >= td_->totalCount) {
        return;
    }

    for (uint64_t i = beginIdx; i < endIdx; ++i) {
        int64_t bufOff = static_cast<int64_t>((i - beginIdx) % BUFFER_NUM) * pingpongOff_;
        DoCopyIn(bufOff, static_cast<int64_t>(i));
        InsertSync(HardEvent::MTE2_MTE3);
        DoCopyOut(bufOff, static_cast<int64_t>(i));
        InsertSync(HardEvent::MTE3_MTE2);
    }
}

} // namespace NsSliceLastDim

#endif
