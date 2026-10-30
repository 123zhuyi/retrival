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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file sinh_v2.h
 * \brief SinhV2 kernel class definition (arch32 - Ascend910B)
 *
 * Computes sinh(x) = (exp(x) - exp(-x)) / 2 for each element.
 * - float32 path: direct computation
 * - float16 path: cast to float32 for computation, then cast back
 *
 * Uses double buffering (BUFFER_NUM=2) for pipeline parallelism:
 * - CopyIn:  GM -> UB (input data transfer)
 * - Compute: UB vector computation (sinh formula)
 * - CopyOut: UB -> GM (output data transfer)
 */
#ifndef SINH_V2_H
#define SINH_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sinh_v2_tiling_data.h"
#include "sinh_v2_tiling_key.h"

namespace NsSinhV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SinhV2 {
public:
    __aicore__ inline SinhV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SinhV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ============================================================================
// Init - Initialize GM pointers and allocate UB buffers
// ============================================================================
template <typename T>
__aicore__ inline void SinhV2<T>::Init(GM_ADDR x, GM_ADDR y, const SinhV2TilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    // Clamp to 0 for idle cores (when totalNum < coreNum, some cores have no work)
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    // Guard: empty tensor or idle core - skip buffer allocation
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    // Temporary buffers for intermediate computation (exp results)
    // For float32: used as float buffers directly
    // For float16: used as float32 buffers for precision-promoted computation
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
}

// ============================================================================
// CopyIn - Transfer data from GM to UB
// ============================================================================
template <typename T>
__aicore__ inline void SinhV2<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueue.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

// ============================================================================
// CopyOut - Transfer data from UB to GM
// ============================================================================
template <typename T>
__aicore__ inline void SinhV2<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> yLocal = outputQueue.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

// ============================================================================
// Compute - float32 specialization: direct sinh computation
// sinh(x) = (exp(x) - exp(-x)) / 2
// ============================================================================
template <>
__aicore__ inline void SinhV2<float>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<float> xLocal = inputQueue.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = outputQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> expPos = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> expNeg = tmpBuf2.Get<float>();

    // Step 1: exp(x)
    AscendC::Exp(expPos, xLocal, currentNum);
    // Step 2: -x
    AscendC::Muls(xLocal, xLocal, static_cast<float>(-1.0f), currentNum);
    // Step 3: exp(-x)
    AscendC::Exp(expNeg, xLocal, currentNum);
    // Step 4: exp(x) - exp(-x)
    AscendC::Sub(expPos, expPos, expNeg, currentNum);
    // Step 5: (exp(x) - exp(-x)) * 0.5
    AscendC::Muls(yLocal, expPos, static_cast<float>(0.5f), currentNum);

    outputQueue.EnQue<float>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

// ============================================================================
// Compute - float16 specialization: cast to float32, compute, cast back
// ============================================================================
template <>
__aicore__ inline void SinhV2<half>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<half> xLocal = inputQueue.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = outputQueue.AllocTensor<half>();
    AscendC::LocalTensor<float> xFloat = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> expNeg = tmpBuf2.Get<float>();

    // Step 1: Cast half -> float for precision
    AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
    // Step 2: exp(x) - store in xLocal's position (we reuse xFloat after saving exp result)
    // We need expPos, so let's compute exp(x) into expNeg first as temp, then swap
    // Actually: compute exp(x) into expNeg (as temp for expPos), then compute -x, then exp(-x)
    AscendC::Exp(expNeg, xFloat, currentNum); // expNeg temporarily holds exp(x)
    // Step 3: -x
    AscendC::Muls(xFloat, xFloat, static_cast<float>(-1.0f), currentNum);
    // Step 4: exp(-x) - store back into xFloat (reuse)
    AscendC::Exp(xFloat, xFloat, currentNum); // xFloat now holds exp(-x)
    // Step 5: exp(x) - exp(-x) : expNeg has exp(x), xFloat has exp(-x)
    AscendC::Sub(expNeg, expNeg, xFloat, currentNum);
    // Step 6: * 0.5
    AscendC::Muls(xFloat, expNeg, static_cast<float>(0.5f), currentNum);
    // Step 7: Cast float -> half
    AscendC::Cast(yLocal, xFloat, AscendC::RoundMode::CAST_ROUND, currentNum);

    outputQueue.EnQue<half>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

// ============================================================================
// Process - Main loop: iterate over tiles
// ============================================================================
template <typename T>
__aicore__ inline void SinhV2<T>::Process()
{
    // Guard: empty tensor or idle core - nothing to process
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsSinhV2
#endif // SINH_V2_H
