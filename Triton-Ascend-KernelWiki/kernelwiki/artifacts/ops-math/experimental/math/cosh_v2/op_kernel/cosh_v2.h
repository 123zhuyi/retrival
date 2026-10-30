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
 * \file cosh_v2.h
 * \brief CoshV2 operator kernel class definition (arch32 / Ascend910B)
 *
 * Computes: cosh(x) = exp(|x| - ln2) + exp(-|x|) / 2
 *
 * This formulation is algebraically equivalent to the standard formula
 * (exp(x) + exp(-x)) / 2, but avoids intermediate overflow:
 *   exp(|x| - ln2) = exp(|x|) / 2, which stays representable whenever
 *   cosh(x) itself is representable.
 *
 * Template parameters:
 *   - T: Data type (half / float / bfloat16_t)
 *   - BUFFER_MODE: Buffer mode (0=single, 1=double buffer)
 *
 * For fp16/bf16: Cast to fp32, compute in fp32, Cast back
 *   (Intermediate fp32 computation ensures both numerical stability
 *    and sufficient precision for the exp(|x|-ln2) formulation)
 */
#ifndef COSH_V2_H
#define COSH_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cosh_v2_tiling_data.h"
#include "cosh_v2_tiling_key.h"

namespace NsCoshV2 {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class CoshV2Op {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;

    // Compute type: fp16 and bf16 compute in float for precision;
    // fp32 computes natively
    using ComputeT = typename std::conditional<std::is_same<T, float>::value, float, float>::type;

public:
    __aicore__ inline CoshV2Op() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, const CoshV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBufExpPos;
    TBuf<QuePosition::VECCALC> tmpBufExpNeg;

    // Extra cast buffer for bf16 path
    TBuf<QuePosition::VECCALC> castBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void CoshV2Op<T, BUFFER_MODE>::Init(GM_ADDR self, GM_ADDR out, const CoshV2TilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    inputGM.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBufExpPos, ubLength_ * sizeof(ComputeT));
    pipe.InitBuffer(tmpBufExpNeg, ubLength_ * sizeof(ComputeT));

    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void CoshV2Op<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> selfLocal = inputQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(selfLocal, inputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void CoshV2Op<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> selfLocal = inputQueue.template DeQue<T>();
    AscendC::LocalTensor<T> outLocal = outputQueue.template AllocTensor<T>();
    AscendC::LocalTensor<ComputeT> expPos = tmpBufExpPos.template Get<ComputeT>();
    AscendC::LocalTensor<ComputeT> expNeg = tmpBufExpNeg.template Get<ComputeT>();

    // Numerically stable formula:
    //   cosh(x) = exp(|x| - ln2) + exp(-|x|) / 2
    //
    // This is algebraically equivalent to (exp(x) + exp(-x)) / 2 because:
    //   exp(|x| - ln2) = exp(|x|) / 2
    // So: exp(|x|)/2 + exp(-|x|)/2 = (exp(|x|) + exp(-|x|))/2 = cosh(x)
    //
    // The key advantage: exp(|x| - ln2) = exp(|x|)/2 <= cosh(x), so it
    // never overflows as long as cosh(x) itself is representable.
    // The naive exp(|x|) can overflow even when cosh(x) is in range.

    constexpr float LN2_F = 0.693147180559945309417232121458176568f;

    if constexpr (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
        // fp16/bf16 path: Cast to float, compute in fp32, Cast back
        // Using fp32 intermediate ensures both precision and numerical stability
        AscendC::LocalTensor<float> castLocal = castBuf.template Get<float>();
        AscendC::LocalTensor<float> fExpPos = tmpBufExpPos.template Get<float>();
        AscendC::LocalTensor<float> fExpNeg = tmpBufExpNeg.template Get<float>();

        // Step 1: Cast input -> fp32
        AscendC::Cast(castLocal, selfLocal, AscendC::RoundMode::CAST_NONE, currentNum);

        // Step 2: |x|
        AscendC::Abs(fExpPos, castLocal, currentNum);

        // Step 3: -|x|
        AscendC::Muls(fExpNeg, fExpPos, -1.0f, currentNum);

        // Step 4: |x| - ln2
        AscendC::Adds(castLocal, fExpPos, -LN2_F, currentNum);

        // Step 5: exp(|x| - ln2) = exp(|x|) / 2
        AscendC::Exp(castLocal, castLocal, currentNum);

        // Step 6: exp(-|x|)
        AscendC::Exp(fExpNeg, fExpNeg, currentNum);

        // Step 7: exp(-|x|) / 2
        AscendC::Muls(fExpNeg, fExpNeg, 0.5f, currentNum);

        // Step 8: exp(|x| - ln2) + exp(-|x|) / 2 = cosh(x)
        AscendC::Add(castLocal, castLocal, fExpNeg, currentNum);

        // Step 9: Cast fp32 -> input dtype
        AscendC::Cast(outLocal, castLocal, AscendC::RoundMode::CAST_ROUND, currentNum);
    } else {
        // fp32 path: compute directly in native precision
        // Step 1: |x|
        AscendC::Abs(expPos, selfLocal, currentNum);

        // Step 2: -|x|
        AscendC::Muls(expNeg, expPos, -1.0f, currentNum);

        // Step 3: |x| - ln2
        AscendC::Adds(outLocal, expPos, -LN2_F, currentNum);

        // Step 4: exp(|x| - ln2) = exp(|x|) / 2
        AscendC::Exp(outLocal, outLocal, currentNum);

        // Step 5: exp(-|x|)
        AscendC::Exp(expNeg, expNeg, currentNum);

        // Step 6: exp(-|x|) / 2
        AscendC::Muls(expNeg, expNeg, 0.5f, currentNum);

        // Step 7: exp(|x| - ln2) + exp(-|x|) / 2 = cosh(x)
        AscendC::Add(outLocal, outLocal, expNeg, currentNum);
    }

    outputQueue.template EnQue<T>(outLocal);
    inputQueue.FreeTensor(selfLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void CoshV2Op<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], outLocal, copyParams);
    outputQueue.FreeTensor(outLocal);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void CoshV2Op<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsCoshV2
#endif // COSH_V2_H
