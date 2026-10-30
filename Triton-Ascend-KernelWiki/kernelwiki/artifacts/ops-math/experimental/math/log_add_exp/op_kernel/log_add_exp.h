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
 * \file log_add_exp.h
 * \brief LogAddExp kernel class definition (arch32)
 *
 * Computes logaddexp(x, y) = max(x, y) + ln(1 + e^(-|x - y|))
 *
 * Uses double buffering (depth=2) for input/output queues.
 * T_IN: input/output data type (float, half, bfloat16_t)
 * T_COMPUTE: computation data type (always float for numerical stability)
 */
#ifndef LOG_ADD_EXP_H
#define LOG_ADD_EXP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log_add_exp_tiling_data.h"
#include "log_add_exp_tiling_key.h"

namespace NsLogAddExp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T_IN, typename T_COMPUTE>
class LogAddExp {
public:
    __aicore__ inline LogAddExp(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR workspace, const LogAddExpTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline int64_t ComputeOffset(int64_t linearIdx, const int64_t* strides, int64_t dimNum);

    // Binary doubling broadcast expansion (called once before the main loop)
    __aicore__ inline void ExpandBroadcast();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;
    TBuf<QuePosition::VECCALC> tmpBuf3;

    GlobalTensor<T_IN> inputGMX;
    GlobalTensor<T_IN> inputGMY;
    GlobalTensor<T_IN> outputGM;
    GlobalTensor<T_IN> workspaceGM; // expanded tensor (binary doubling)

    // Raw GM addresses for ExpandBroadcast (to allow type reinterpretation)
    GM_ADDR rawExpandSrc_ = nullptr;
    GM_ADDR rawWorkspace_ = nullptr;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    const LogAddExpTilingData* tilingData_ = nullptr;
};

template <typename T_IN, typename T_COMPUTE>
__aicore__ inline void LogAddExp<T_IN, T_COMPUTE>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR workspace, const LogAddExpTilingData* tilingData)
{
    tilingData_ = tilingData;
    int64_t remainderLength = tilingData->totalLength - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    // EXPERIMENT 12a: RE-ENABLE BD Init branch, keep ExpandBroadcast skipped
    if (tilingData->useBinaryDoubling) {
        // Modular broadcast path: linear tensor uses per-core offset,
        // broadcast tensor (1,N) uses base address with no core offset
        int64_t coreOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
        if (tilingData->expandSrcIsX) {
            // x is (1,N) broadcast, y is (M,N) linear
            inputGMX.SetGlobalBuffer((__gm__ T_IN*)x, tilingData->innerSize);
            inputGMY.SetGlobalBuffer((__gm__ T_IN*)y + coreOffset, blockLength_);
        } else {
            // y is (1,N) broadcast, x is (M,N) linear
            inputGMX.SetGlobalBuffer((__gm__ T_IN*)x + coreOffset, blockLength_);
            inputGMY.SetGlobalBuffer((__gm__ T_IN*)y, tilingData->innerSize);
        }
    } else if (tilingData->needBroadcast == 0) {
        // Non-broadcast path: use offset-based addressing
        inputGMX.SetGlobalBuffer((__gm__ T_IN*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
        inputGMY.SetGlobalBuffer((__gm__ T_IN*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    } else {
        // General broadcast path: full base address for both, offset computed per element
        inputGMX.SetGlobalBuffer((__gm__ T_IN*)x);
        inputGMY.SetGlobalBuffer((__gm__ T_IN*)y);
    }
    outputGM.SetGlobalBuffer((__gm__ T_IN*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // Buffer size must be 32-byte aligned for DataCopy/DataCopyPad
    constexpr int64_t ALIGN_BYTES = 32;
    constexpr int64_t ALIGN_ELEMS_IN = ALIGN_BYTES / sizeof(T_IN);
    constexpr int64_t ALIGN_ELEMS_COMPUTE = ALIGN_BYTES / sizeof(T_COMPUTE);
    int64_t alignedUbIn = ((ubLength_ + ALIGN_ELEMS_IN - 1) / ALIGN_ELEMS_IN) * ALIGN_ELEMS_IN;
    int64_t alignedUbCompute = ((ubLength_ + ALIGN_ELEMS_COMPUTE - 1) / ALIGN_ELEMS_COMPUTE) * ALIGN_ELEMS_COMPUTE;

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, alignedUbIn * sizeof(T_IN));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, alignedUbIn * sizeof(T_IN));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, alignedUbIn * sizeof(T_IN));
    // Temp buffers for intermediate fp32 computations
    pipe.InitBuffer(tmpBuf1, alignedUbCompute * sizeof(T_COMPUTE));
    pipe.InitBuffer(tmpBuf2, alignedUbCompute * sizeof(T_COMPUTE));
    pipe.InitBuffer(tmpBuf3, alignedUbCompute * sizeof(T_COMPUTE));
}

// -----------------------------------------------------------------------
// ExpandBroadcast: binary doubling pre-expansion (binary doubling path only)
//
// Each core independently expands its row slice into workspace:
//   step0: copy src[0:N] → ws[rStart*N] through UB  (1 row)
//   then binary doubling: ws[rStart:rStart+cur] → ws[rStart+cur:rStart+2*cur]
//   until all rows [rStart, rEnd] are filled.
//
// After this, workspaceGM[blockFactor*blockIdx + k] == src[k % N]
// for all k in [0, blockLength_), enabling sequential DataCopy in CopyIn.
// -----------------------------------------------------------------------
template <typename T_IN, typename T_COMPUTE>
__aicore__ inline void LogAddExp<T_IN, T_COMPUTE>::ExpandBroadcast()
{
    // No longer needed - broadcast handled in CopyIn via modular addressing
}

template <typename T_IN, typename T_COMPUTE>
__aicore__ inline int64_t LogAddExp<T_IN, T_COMPUTE>::ComputeOffset(
    int64_t linearIdx, const int64_t* strides, int64_t dimNum)
{
    int64_t offset = 0;
    int64_t remaining = linearIdx;

    // Decompose linear index to multi-dimensional indices
    for (int64_t i = 0; i < dimNum; i++) {
        int64_t dimIdx;
        if (i == dimNum - 1) {
            dimIdx = remaining;
        } else {
            int64_t dimStride = tilingData_->outStrides[i];
            dimIdx = remaining / dimStride;
            remaining = remaining % dimStride;
        }
        offset += dimIdx * strides[i];
    }

    return offset;
}

template <typename T_IN, typename T_COMPUTE>
__aicore__ inline void LogAddExp<T_IN, T_COMPUTE>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T_IN> xLocal = inputQueueX.AllocTensor<T_IN>();
    AscendC::LocalTensor<T_IN> yLocal = inputQueueY.AllocTensor<T_IN>();

    if (tilingData_->useBinaryDoubling) {
        // Modular broadcast CopyIn for (1,N) + (M,N) case
        // The non-broadcast tensor (y if expandSrcIsX=1, x if expandSrcIsX=0) is linear
        // The broadcast tensor (x if expandSrcIsX=1, y if expandSrcIsX=0) needs x[globalIdx % N]
        int64_t offset = progress * ubLength_;
        int64_t N = tilingData_->innerSize;
        int64_t blockStart = tilingData_->blockFactor * AscendC::GetBlockIdx();
        int64_t globalStart = blockStart + offset;
        int64_t modStart = globalStart % N; // where we are within the period

        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        // Linear tensor: simple sequential copy
        copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T_IN));
        if (tilingData_->expandSrcIsX) {
            // y is linear (M,N), x is broadcast (1,N)
            AscendC::DataCopyPad(yLocal, inputGMY[offset], copyParams, {false, 0, 0, 0});
        } else {
            // x is linear (M,N), y is broadcast (1,N)
            AscendC::DataCopyPad(xLocal, inputGMX[offset], copyParams, {false, 0, 0, 0});
        }

        // Broadcast tensor: modular copy from the (1,N) tensor
        // The broadcast GM tensor was set up in Init with base address (no core offset)
        AscendC::LocalTensor<T_IN>& bcastLocal = tilingData_->expandSrcIsX ? xLocal : yLocal;
        // Select the correct GM tensor for broadcast source
        AscendC::GlobalTensor<T_IN>& bcastGM = tilingData_->expandSrcIsX ? inputGMX : inputGMY;

        // Fill bcastLocal[0..currentNum-1] with bcastGM[modStart], wrapping at N
        // Multiple DataCopyPad calls to non-overlapping UB regions: no PipeBarrier needed
        int64_t filled = 0;

        // First chunk: from modStart to min(N, modStart + currentNum)
        {
            int64_t chunk = N - modStart;
            if (chunk > currentNum)
                chunk = currentNum;
            copyParams.blockLen = static_cast<uint16_t>(chunk * sizeof(T_IN));
            AscendC::DataCopyPad(bcastLocal, bcastGM[modStart], copyParams, {false, 0, 0, 0});
            filled = chunk;
        }

        // Full period copies from bcastGM[0], N elements each
        while (filled + N <= currentNum) {
            copyParams.blockLen = static_cast<uint16_t>(N * sizeof(T_IN));
            AscendC::DataCopyPad(bcastLocal[filled], bcastGM[0], copyParams, {false, 0, 0, 0});
            filled += N;
        }

        // Last partial chunk (if any)
        if (filled < currentNum) {
            int64_t remainder = currentNum - filled;
            copyParams.blockLen = static_cast<uint16_t>(remainder * sizeof(T_IN));
            AscendC::DataCopyPad(bcastLocal[filled], bcastGM[0], copyParams, {false, 0, 0, 0});
        }

    } else if (tilingData_->needBroadcast == 0) {
        // Non-broadcast path: sequential DataCopyPad
        int64_t offset = progress * ubLength_;

        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T_IN));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopyPad(xLocal, inputGMX[offset], copyParams, {false, 0, 0, 0});
        AscendC::DataCopyPad(yLocal, inputGMY[offset], copyParams, {false, 0, 0, 0});
    } else {
        // General broadcast path: element-by-element copy with index calculation
        int64_t blockOffset = tilingData_->blockFactor * AscendC::GetBlockIdx();
        int64_t tileOffset = progress * ubLength_;

        for (int64_t i = 0; i < currentNum; i++) {
            int64_t globalIdx = blockOffset + tileOffset + i;
            int64_t xOffset = ComputeOffset(globalIdx, tilingData_->xStrides, tilingData_->dimNum);
            int64_t yOffset = ComputeOffset(globalIdx, tilingData_->yStrides, tilingData_->dimNum);
            xLocal.SetValue(i, inputGMX.GetValue(xOffset));
            yLocal.SetValue(i, inputGMY.GetValue(yOffset));
        }
    }

    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

template <typename T_IN, typename T_COMPUTE>
__aicore__ inline void LogAddExp<T_IN, T_COMPUTE>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T_IN> outLocal = outputQueue.DeQue<T_IN>();

    int64_t offset = progress * ubLength_;

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentNum * sizeof(T_IN));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[offset], outLocal, copyParams);

    outputQueue.FreeTensor(outLocal);
}

template <typename T_IN, typename T_COMPUTE>
__aicore__ inline void LogAddExp<T_IN, T_COMPUTE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T_IN> xLocal = inputQueueX.DeQue<T_IN>();
    AscendC::LocalTensor<T_IN> yLocal = inputQueueY.DeQue<T_IN>();
    AscendC::LocalTensor<T_IN> outLocal = outputQueue.AllocTensor<T_IN>();

    AscendC::LocalTensor<T_COMPUTE> tmp1 = tmpBuf1.Get<T_COMPUTE>();
    AscendC::LocalTensor<T_COMPUTE> tmp2 = tmpBuf2.Get<T_COMPUTE>();

    if constexpr (std::is_same_v<T_IN, T_COMPUTE>) {
        // fp32 path: no type conversion needed
        // logaddexp(x, y) = max(x, y) + ln(1 + e^(-|x - y|))

        AscendC::Max(tmp2, xLocal, yLocal, currentNum);
        AscendC::Sub(tmp1, xLocal, yLocal, currentNum);
        AscendC::Abs(tmp1, tmp1, currentNum);
        AscendC::Muls(tmp1, tmp1, static_cast<T_COMPUTE>(-1.0f), currentNum);
        AscendC::Exp(tmp1, tmp1, currentNum);
        AscendC::Adds(tmp1, tmp1, static_cast<T_COMPUTE>(1.0f), currentNum);
        AscendC::Ln(tmp1, tmp1, currentNum);
        AscendC::Add(outLocal, tmp2, tmp1, currentNum);

        // Fix inf + nan = inf case (when both inputs are inf/-inf)
        constexpr T_COMPUTE POS_INF = __builtin_huge_valf();
        constexpr T_COMPUTE NEG_INF = -__builtin_huge_valf();

        for (int64_t i = 0; i < currentNum; i++) {
            T_COMPUTE maxVal = tmp2.GetValue(i);
            T_COMPUTE outVal = outLocal.GetValue(i);
            bool isMaxInf = (maxVal == POS_INF || maxVal == NEG_INF);
            bool isOutNan = (outVal != outVal);
            if (isMaxInf && isOutNan) {
                outLocal.SetValue(i, static_cast<T_IN>(maxVal));
            }
        }
    } else {
        // fp16/bf16 path: cast to fp32, compute, cast back
        AscendC::LocalTensor<T_COMPUTE> tmp3 = tmpBuf3.Get<T_COMPUTE>();

        AscendC::Cast(tmp1, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::Cast(tmp3, yLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::Max(tmp2, tmp1, tmp3, currentNum);
        AscendC::Sub(tmp1, tmp1, tmp3, currentNum);
        AscendC::Abs(tmp1, tmp1, currentNum);
        AscendC::Muls(tmp1, tmp1, static_cast<T_COMPUTE>(-1.0f), currentNum);
        AscendC::Exp(tmp1, tmp1, currentNum);
        AscendC::Adds(tmp1, tmp1, static_cast<T_COMPUTE>(1.0f), currentNum);
        AscendC::Ln(tmp1, tmp1, currentNum);
        AscendC::Add(tmp2, tmp2, tmp1, currentNum);
        AscendC::Cast(outLocal, tmp2, AscendC::RoundMode::CAST_ROUND, currentNum);
    }

    outputQueue.EnQue<T_IN>(outLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

template <typename T_IN, typename T_COMPUTE>
__aicore__ inline void LogAddExp<T_IN, T_COMPUTE>::Process()
{
    // Modular broadcast: no pre-expansion needed, broadcast handled in CopyIn
    // ExpandBroadcast is now empty and not needed

    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsLogAddExp
#endif // LOG_ADD_EXP_H
