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
 * \file tan_v3.h
 * \brief TanV3 kernel class definition (arch32 - Ascend910B)
 *
 * Computes tan(x) = sin(x) / cos(x) for each element.
 * - float32 path: direct computation using Sin, Cos, Div
 * - float16 path: cast to float32 for computation, then cast back
 *
 * Uses double buffering (BUFFER_NUM=2) for pipeline parallelism:
 * - CopyIn:  GM -> UB (input data transfer)
 * - Compute: UB vector computation (tan formula)
 * - CopyOut: UB -> GM (output data transfer)
 */
#ifndef TAN_V3_H
#define TAN_V3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tan_v3_tiling_data.h"
#include "tan_v3_tiling_key.h"

namespace NsTanV3 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TanV3 {
public:
    __aicore__ inline TanV3(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TanV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1; // stores sin(x) intermediate result
    TBuf<QuePosition::VECCALC> tmpBuf2; // stores cos(x) intermediate result

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ============================================================================
// Init - Initialize GM pointers and allocate UB buffers
// ============================================================================
template <typename T>
__aicore__ inline void TanV3<T>::Init(GM_ADDR x, GM_ADDR y, const TanV3TilingData* tilingData)
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
    // Temporary buffers for sin and cos intermediate results
    // For float32: used as float buffers directly
    // For float16: used as float32 buffers for precision-promoted computation
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
}

// ============================================================================
// CopyIn - Transfer data from GM to UB
// ============================================================================
template <typename T>
__aicore__ inline void TanV3<T>::CopyIn(int64_t progress, int64_t currentNum)
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
__aicore__ inline void TanV3<T>::CopyOut(int64_t progress, int64_t currentNum)
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
// Compute - float32 specialization: direct tan computation
// tan(x) = sin(x) / cos(x)
// ============================================================================
template <>
__aicore__ inline void TanV3<float>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<float> xLocal = inputQueue.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = outputQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> sinVal = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> cosVal = tmpBuf2.Get<float>();

    // Step 1: sin(x) -> sinVal (tmpBuf1)
    AscendC::Sin(sinVal, xLocal, currentNum);
    // Step 2: cos(x) -> cosVal (tmpBuf2)
    AscendC::Cos(cosVal, xLocal, currentNum);
    // Step 3: sin(x) / cos(x) -> yLocal
    AscendC::Div(yLocal, sinVal, cosVal, currentNum);

    outputQueue.EnQue<float>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

// ============================================================================
// Compute - float16 specialization: cast to float32, compute, cast back
// Flow: Cast(half->float) -> Cos -> Sin -> Div -> Cast(float->half)
// Key constraint: Cos must be computed before Sin to avoid overwriting input
// ============================================================================
template <>
__aicore__ inline void TanV3<half>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<half> xLocal = inputQueue.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = outputQueue.AllocTensor<half>();
    AscendC::LocalTensor<float> sinVal = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> cosVal = tmpBuf2.Get<float>();

    // Step 1: Cast half -> float (store in sinVal as temp for x_float)
    AscendC::Cast(sinVal, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
    // Step 2: cos(x) - must compute before sin overwrites sinVal
    AscendC::Cos(cosVal, sinVal, currentNum);
    // Step 3: sin(x) - overwrites sinVal (cosVal already saved)
    AscendC::Sin(sinVal, sinVal, currentNum);
    // Step 4: sin(x) / cos(x) -> sinVal (reuse)
    AscendC::Div(sinVal, sinVal, cosVal, currentNum);
    // Step 5: Cast float -> half
    AscendC::Cast(yLocal, sinVal, AscendC::RoundMode::CAST_ROUND, currentNum);

    outputQueue.EnQue<half>(yLocal);
    inputQueue.FreeTensor(xLocal);
}

// ============================================================================
// Process - Main loop: iterate over tiles
// ============================================================================
template <typename T>
__aicore__ inline void TanV3<T>::Process()
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

} // namespace NsTanV3
#endif // TAN_V3_H
