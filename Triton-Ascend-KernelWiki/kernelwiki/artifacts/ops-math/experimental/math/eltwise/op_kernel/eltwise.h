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
 * \file eltwise.h
 * \brief Eltwise kernel class definition (arch35)
 *
 * Supports 3 modes: PRODUCT(0), SUM(1), MAX(2)
 * Supports 3 dtypes: float16, bfloat16, float32
 *
 * Precision strategy:
 *   - float32: compute directly (no Cast needed)
 *   - float16/bfloat16: Cast to fp32 -> compute -> Cast back
 *
 * DYNAMIC_INPUT handling:
 *   The "inputs" GM_ADDR is passed in folded_with_desc format by the framework.
 *   Use AscendC::ListTensorDesc to parse it and retrieve individual tensor
 *   data pointers via GetDataPtr<T>(index).
 *
 * Buffer layout (single buffer, simplified):
 *   For FP32 (no cast needed):
 *     inputBuf (VECIN, depth=1):  ubFactor * sizeof(float)
 *     accBuf   (VECCALC):         ubFactor * sizeof(float)
 *     outputBuf(VECOUT, depth=1): ubFactor * sizeof(float)
 *     Total: 3 * ubFactor * 4 bytes
 *
 *   For FP16/BF16 (cast needed):
 *     inputBuf (VECIN, depth=1):  ubFactor * sizeof(T)      -- raw input
 *     castBuf  (VECCALC):         ubFactor * sizeof(float)   -- casted input
 *     accBuf   (VECCALC):         ubFactor * sizeof(float)   -- accumulator
 *     outputBuf(VECOUT, depth=1): ubFactor * sizeof(T)       -- cast-back output
 *     Total: ubFactor * (2*sizeof(T) + 2*sizeof(float)) bytes
 */
#ifndef ELTWISE_H
#define ELTWISE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "basic_api/kernel_operator_list_tensor_intf.h"
#include "eltwise_tiling_data.h"
#include "eltwise_tiling_key.h"

namespace NsEltwise {

constexpr uint32_t MAX_INPUT_NUM = 32;

using AscendC::Add;
using AscendC::Cast;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::Duplicate;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Max;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename T, int MODE>
class Eltwise {
public:
    __aicore__ inline Eltwise() {}

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const EltwiseTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t inputIdx, int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t gmOffset, int64_t currentNum);

    // Compute a single tile: iterate over all inputs, accumulate result
    __aicore__ inline void ComputeTile(int64_t gmOffset, int64_t currentNum);

    // Mode-specific computation helpers (all operate in fp32)
    __aicore__ inline void ComputeProductInit(LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum);
    __aicore__ inline void ComputeProductAccum(LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum);
    __aicore__ inline void ComputeSumInit(
        LocalTensor<float>& acc, LocalTensor<float>& src, float coeffVal, int64_t alignedNum);
    __aicore__ inline void ComputeSumAccum(
        LocalTensor<float>& acc, LocalTensor<float>& src, float coeffVal, int64_t alignedNum);
    __aicore__ inline void ComputeMaxInit(LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum);
    __aicore__ inline void ComputeMaxAccum(LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> castBuf_; // cast buffer (fp32) for non-fp32 dtypes
    TBuf<QuePosition::VECCALC> accBuf_;  // accumulator buffer (fp32)

    GlobalTensor<T> inputGM_[MAX_INPUT_NUM];
    GlobalTensor<T> outputGM_;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
    uint32_t inputNum_ = 0;
    float coeff_[MAX_INPUT_NUM] = {0.0f};
};

// =============================================================================
// Init
// =============================================================================
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::Init(GM_ADDR inputs, GM_ADDR output, const EltwiseTilingData* tilingData)
{
    ubFactor_ = tilingData->ubFactor;
    inputNum_ = tilingData->inputNum;

    // Copy coeff values
    for (uint32_t i = 0; i < MAX_INPUT_NUM; i++) {
        coeff_[i] = tilingData->coeff[i];
    }

    if (tilingData->totalNum == 0 || tilingData->blockFactor == 0 || inputNum_ == 0) {
        blockOffset_ = 0;
        blockLen_ = 0;
        return;
    }

    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remaining = tilingData->totalNum - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;

    // DYNAMIC_INPUT: use ListTensorDesc to parse folded_with_desc format
    AscendC::ListTensorDesc inputListDesc(reinterpret_cast<__gm__ void*>(inputs));
    for (uint32_t i = 0; i < inputNum_; i++) {
        __gm__ T* tensorAddr = inputListDesc.GetDataPtr<T>(i);
        inputGM_[i].SetGlobalBuffer(tensorAddr + blockOffset_, blockLen_);
    }
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output) + blockOffset_, blockLen_);

    // Initialize buffers
    pipe_.InitBuffer(inputQueue_, 1, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, 1, ubFactor_ * sizeof(T));

    if constexpr (std::is_same_v<T, float>) {
        // FP32: accBuf only (no cast needed)
        pipe_.InitBuffer(accBuf_, ubFactor_ * sizeof(float));
    } else {
        // FP16/BF16: need castBuf and accBuf (both fp32)
        pipe_.InitBuffer(castBuf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(accBuf_, ubFactor_ * sizeof(float));
    }
}

// =============================================================================
// CopyIn: GM -> UB (inputQueue)
// =============================================================================
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::CopyIn(uint32_t inputIdx, int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueue_.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, inputGM_[inputIdx][gmOffset], copyParams, {false, 0, 0, 0});
    inputQueue_.EnQue(xLocal);
}

// =============================================================================
// CopyOut: UB -> GM (outputQueue)
// =============================================================================
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::CopyOut(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueue_.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outputGM_[gmOffset], yLocal, copyParams);
    outputQueue_.FreeTensor(yLocal);
}

// =============================================================================
// Mode-specific computation helpers (all in fp32 space)
// =============================================================================

// PRODUCT: Initialize accumulator = src (first input)
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeProductInit(
    LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum)
{
    // NOTE: Duplicate(0)+Add used as cross-queue tensor copy workaround
    Duplicate(acc, 0.0f, static_cast<int32_t>(alignedNum));
    Add(acc, acc, src, static_cast<int32_t>(alignedNum));
}

// PRODUCT: acc = acc * src
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeProductAccum(
    LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum)
{
    Mul(acc, acc, src, static_cast<int32_t>(alignedNum));
}

// SUM: Initialize accumulator = coeff * src (first input)
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeSumInit(
    LocalTensor<float>& acc, LocalTensor<float>& src, float coeffVal, int64_t alignedNum)
{
    Muls(acc, src, coeffVal, static_cast<int32_t>(alignedNum));
}

// SUM: acc = acc + coeff * src
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeSumAccum(
    LocalTensor<float>& acc, LocalTensor<float>& src, float coeffVal, int64_t alignedNum)
{
    // Muls into src itself (overwrite ok, we don't need original after this)
    Muls(src, src, coeffVal, static_cast<int32_t>(alignedNum));
    Add(acc, acc, src, static_cast<int32_t>(alignedNum));
}

// MAX: Initialize accumulator = src (first input)
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeMaxInit(
    LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum)
{
    // NOTE: Duplicate(0)+Add used as cross-queue tensor copy workaround
    Duplicate(acc, 0.0f, static_cast<int32_t>(alignedNum));
    Add(acc, acc, src, static_cast<int32_t>(alignedNum));
}

// MAX: acc = max(acc, src)
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeMaxAccum(
    LocalTensor<float>& acc, LocalTensor<float>& src, int64_t alignedNum)
{
    Max(acc, acc, src, static_cast<int32_t>(alignedNum));
}

// =============================================================================
// ComputeTile: process one tile, iterating over all inputs
// =============================================================================
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::ComputeTile(int64_t gmOffset, int64_t currentNum)
{
    // Alignment: max of float32 block and T block (32-byte boundary)
    constexpr int64_t floatBlock = 32 / sizeof(float); // 8
    constexpr int64_t typeBlock = 32 / sizeof(T);
    constexpr int64_t alignBlock = (floatBlock > typeBlock) ? floatBlock : typeBlock;
    int64_t alignedNum = ((currentNum + alignBlock - 1) / alignBlock) * alignBlock;

    LocalTensor<float> acc = accBuf_.template Get<float>();

    // Process first input: initialize accumulator
    {
        CopyIn(0, gmOffset, currentNum);
        LocalTensor<T> xLocal = inputQueue_.template DeQue<T>();

        if constexpr (std::is_same_v<T, float>) {
            // FP32: direct computation
            if constexpr (MODE == 0) {
                ComputeProductInit(acc, xLocal, alignedNum);
            } else if constexpr (MODE == 1) {
                ComputeSumInit(acc, xLocal, coeff_[0], alignedNum);
            } else {
                ComputeMaxInit(acc, xLocal, alignedNum);
            }
        } else {
            // FP16/BF16: cast to fp32 first
            LocalTensor<float> castLocal = castBuf_.template Get<float>();
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));

            if constexpr (MODE == 0) {
                ComputeProductInit(acc, castLocal, alignedNum);
            } else if constexpr (MODE == 1) {
                ComputeSumInit(acc, castLocal, coeff_[0], alignedNum);
            } else {
                ComputeMaxInit(acc, castLocal, alignedNum);
            }
        }
        inputQueue_.FreeTensor(xLocal);
    }

    // Process remaining inputs: accumulate
    for (uint32_t k = 1; k < inputNum_; k++) {
        CopyIn(k, gmOffset, currentNum);
        LocalTensor<T> xLocal = inputQueue_.template DeQue<T>();

        if constexpr (std::is_same_v<T, float>) {
            if constexpr (MODE == 0) {
                ComputeProductAccum(acc, xLocal, alignedNum);
            } else if constexpr (MODE == 1) {
                ComputeSumAccum(acc, xLocal, coeff_[k], alignedNum);
            } else {
                ComputeMaxAccum(acc, xLocal, alignedNum);
            }
        } else {
            LocalTensor<float> castLocal = castBuf_.template Get<float>();
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedNum));

            if constexpr (MODE == 0) {
                ComputeProductAccum(acc, castLocal, alignedNum);
            } else if constexpr (MODE == 1) {
                ComputeSumAccum(acc, castLocal, coeff_[k], alignedNum);
            } else {
                ComputeMaxAccum(acc, castLocal, alignedNum);
            }
        }
        inputQueue_.FreeTensor(xLocal);
    }

    // Write result to output
    LocalTensor<T> yLocal = outputQueue_.template AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        // FP32: acc is already the result, copy to output
        Duplicate(yLocal, 0.0f, static_cast<int32_t>(alignedNum));
        Add(yLocal, yLocal, acc, static_cast<int32_t>(alignedNum));
    } else {
        // FP16/BF16: cast back from fp32
        Cast(yLocal, acc, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedNum));
    }
    outputQueue_.template EnQue<T>(yLocal);
}

// =============================================================================
// Process: main loop over UB chunks
// =============================================================================
template <typename T, int MODE>
__aicore__ inline void Eltwise<T, MODE>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t gmOffset = i * ubFactor_;
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLen_ - gmOffset) : ubFactor_;
        ComputeTile(gmOffset, currentNum);
        CopyOut(gmOffset, currentNum);
    }
}

} // namespace NsEltwise
#endif // ELTWISE_H
