/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file inv.h
 * \brief Inv kernel class definition (arch35)
 *
 * Inv(x) = 1 / x
 *
 * Precision strategy:
 *   - float32: Div(1.0, x) directly (no Reciprocal -- precision requirement)
 *   - float16/bfloat16: Cast to float32 -> Div(1.0, x) -> Cast back
 *
 * Template parameter T (mapped by TilingKey):
 *   - float:       TilingKey 0: direct computation in float32
 *   - half:        TilingKey 1: cast to fp32 -> Div -> cast back to fp16
 *   - bfloat16_t:  TilingKey 2: cast to fp32 -> Div -> cast back to bf16
 *
 * Buffer layout (single buffer):
 *   inputQueue(1 buf):  ubFactor * sizeof(T)
 *   outputQueue(1 buf): ubFactor * sizeof(T)
 *   tmpBuf1_:           ubFactor * sizeof(float)  -- xFloat32 intermediate
 *   tmpBuf2_:           ubFactor * sizeof(float)  -- ones vector
 */
#ifndef INV_H
#define INV_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inv_tiling_data.h"
#include "inv_tiling_key.h"

namespace NsInv {

using AscendC::Cast;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename T>
class Inv {
public:
    __aicore__ inline Inv() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const InvTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);

    // float32 direct path: Div(ones, x)
    __aicore__ inline void ComputeFloat32(LocalTensor<float>& xLocal, LocalTensor<float>& yLocal, int64_t alignedNum);
    // Non-float32: Cast -> Div -> Cast
    template <typename SrcT>
    __aicore__ inline void ComputeWithCast(
        LocalTensor<SrcT>& xLocal, LocalTensor<SrcT>& yLocal, int64_t currentNum, int64_t alignedNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1_; // xFloat32 intermediate / Div result
    TBuf<QuePosition::VECCALC> tmpBuf2_; // ones vector

    GlobalTensor<T> selfGM_;
    GlobalTensor<T> outGM_;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
};

// =============================================================================
// Init
// =============================================================================
template <typename T>
__aicore__ inline void Inv<T>::Init(GM_ADDR self, GM_ADDR out, const InvTilingData* tilingData)
{
    ubFactor_ = tilingData->ubFactor;

    if (tilingData->totalElements == 0 || tilingData->blockFactor == 0) {
        blockOffset_ = 0;
        blockLen_ = 0;
        return;
    }

    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remaining = tilingData->totalElements - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;

    selfGM_.SetGlobalBuffer((__gm__ T*)self + blockOffset_, blockLen_);
    outGM_.SetGlobalBuffer((__gm__ T*)out + blockOffset_, blockLen_);

    pipe.InitBuffer(inputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(outputQueue, 1, ubFactor_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1_, ubFactor_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2_, ubFactor_ * sizeof(float));
}

// =============================================================================
// CopyIn: GM -> UB
// =============================================================================
template <typename T>
__aicore__ inline void Inv<T>::CopyIn(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, selfGM_[gmOffset], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

// =============================================================================
// CopyOut: UB -> GM
// =============================================================================
template <typename T>
__aicore__ inline void Inv<T>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outGM_[gmOffset], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

// =============================================================================
// ComputeFloat32: Direct Div(1.0, x) for float32
// =============================================================================
template <typename T>
__aicore__ inline void Inv<T>::ComputeFloat32(
    LocalTensor<float>& xLocal, LocalTensor<float>& yLocal, int64_t alignedNum)
{
    LocalTensor<float> ones = tmpBuf2_.template Get<float>();
    Duplicate(ones, 1.0f, static_cast<int32_t>(alignedNum));
    Div(yLocal, ones, xLocal, static_cast<int32_t>(alignedNum));
}

// =============================================================================
// ComputeWithCast: Cast(T->f32) -> Div(1.0, x) -> Cast(f32->T)
// =============================================================================
template <typename T>
template <typename SrcT>
__aicore__ inline void Inv<T>::ComputeWithCast(
    LocalTensor<SrcT>& xLocal, LocalTensor<SrcT>& yLocal, int64_t currentNum, int64_t alignedNum)
{
    LocalTensor<float> xFloat = tmpBuf1_.template Get<float>();
    LocalTensor<float> ones = tmpBuf2_.template Get<float>();

    // Step 1: Cast input to float32
    Cast(xFloat, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));

    // Step 2: Generate ones vector and compute Div(1.0, x)
    Duplicate(ones, 1.0f, static_cast<int32_t>(alignedNum));
    Div(xFloat, ones, xFloat, static_cast<int32_t>(alignedNum));

    // Step 3: Cast back to original type
    Cast(yLocal, xFloat, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedNum));
}

// =============================================================================
// Compute: dispatch to float32 direct or cast path
// =============================================================================
template <typename T>
__aicore__ inline void Inv<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue.template DeQue<T>();
    LocalTensor<T> yLocal = outputQueue.template AllocTensor<T>();

    // Align to max of float32 block and T block (32-byte boundary)
    constexpr int64_t floatBlock = 32 / sizeof(float); // 8
    constexpr int64_t typeBlock = 32 / sizeof(T);
    constexpr int64_t alignBlock = (floatBlock > typeBlock) ? floatBlock : typeBlock;
    int64_t alignedNum = ((currentNum + alignBlock - 1) / alignBlock) * alignBlock;

    if constexpr (std::is_same_v<T, float>) {
        ComputeFloat32(xLocal, yLocal, alignedNum);
    } else {
        ComputeWithCast(xLocal, yLocal, currentNum, alignedNum);
    }

    outputQueue.template EnQue<T>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

// =============================================================================
// Process: main loop over UB chunks
// =============================================================================
template <typename T>
__aicore__ inline void Inv<T>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t gmOffset = i * ubFactor_;
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLen_ - gmOffset) : ubFactor_;
        CopyIn(gmOffset, currentNum);
        Compute(currentNum);
        CopyOut(gmOffset, currentNum);
    }
}

} // namespace NsInv
#endif // INV_H
