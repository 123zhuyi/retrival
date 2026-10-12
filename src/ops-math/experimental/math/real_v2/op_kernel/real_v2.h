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

/*!
 * \file real_v2.h
 * \brief RealV2 operator kernel class definition
 *
 * Template parameters:
 *   - T: Output data type (float/half)
 *   - IS_COMPLEX: 0=real passthrough, 1=complex extract real part
 *
 * Passthrough path (IS_COMPLEX=0) uses TBuf<VECCALC>
 * with direct GM->UB->GM copy, completely avoiding UB-to-UB DataCopy
 * and its uint16_t blockLen limitation. All transfers use DataCopyPad
 * with DataCopyExtParams (uint32_t blockLen).
 */
#ifndef REAL_V2_H
#define REAL_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "real_v2_tiling_data.h"
#include "real_v2_tiling_key.h"

namespace NsRealV2 {

using namespace AscendC;

template <typename T, int IS_COMPLEX>
class RealV2Op {
    static constexpr int32_t BUFFER_NUM = 1;

public:
    __aicore__ inline RealV2Op(){};

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const RealV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void BuildOffsetVector();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> passBuf; // Used for passthrough direct copy
    TBuf<QuePosition::VECCALC> offsetBuf;

    GlobalTensor<T> inputGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int IS_COMPLEX>
__aicore__ inline void RealV2Op<T, IS_COMPLEX>::BuildOffsetVector()
{
    // Build Gather offset vector: byte offsets of real parts within interleaved complex data
    // For complex data [real0, imag0, real1, imag1, ...], the byte offsets of real parts are:
    // [0, 2*sizeof(T), 4*sizeof(T), 6*sizeof(T), ...]
    LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
    for (int64_t i = 0; i < ubLength_; i++) {
        offsetLocal.SetValue(i, static_cast<uint32_t>(i * 2 * sizeof(T)));
    }
}

template <typename T, int IS_COMPLEX>
__aicore__ inline void RealV2Op<T, IS_COMPLEX>::Init(
    GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const RealV2TilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalOutputNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    if constexpr (IS_COMPLEX == 0) {
        // Real passthrough: use TBuf<VECCALC> for direct GM->UB->GM copy.
        inputGM.SetGlobalBuffer((__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
        outputGM.SetGlobalBuffer((__gm__ T*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

        pipe.InitBuffer(passBuf, ubLength_ * sizeof(T));
    } else {
        // Complex extraction: input GM contains interleaved [real, imag] pairs as T elements.
        inputGM.SetGlobalBuffer(
            (__gm__ T*)self + tilingData->blockFactor * AscendC::GetBlockIdx() * 2, blockLength_ * 2);
        outputGM.SetGlobalBuffer((__gm__ T*)out + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

        // Input buffer: holds complex data (2x output elements)
        pipe.InitBuffer(inputQueue, BUFFER_NUM, ubLength_ * 2 * sizeof(T));
        // Output buffer: holds extracted real parts
        pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        // Offset buffer for Gather: one uint32_t per output element
        pipe.InitBuffer(offsetBuf, ubLength_ * sizeof(uint32_t));

        // Build offset vector once during init (reused for all UB iterations)
        BuildOffsetVector();
    }
}

template <typename T, int IS_COMPLEX>
__aicore__ inline void RealV2Op<T, IS_COMPLEX>::CopyIn(int64_t progress, int64_t currentNum)
{
    if constexpr (IS_COMPLEX == 0) {
        // Real passthrough: copy GM to TBuf directly
        AscendC::LocalTensor<T> local = passBuf.Get<T>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(
            local, inputGM[progress * ubLength_], copyParams, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        // Wait for MTE2 to complete before MTE3 writes out
        AscendC::PipeBarrier<PIPE_ALL>();
    } else {
        // Complex extraction: copy entire complex data (2x elements) from GM to UB
        AscendC::LocalTensor<T> inLocal = inputQueue.template AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * 2 * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(
            inLocal, inputGM[progress * ubLength_ * 2], copyParams, AscendC::DataCopyPadExtParams<T>{false, 0, 0, 0});
        inputQueue.EnQue(inLocal);
    }
}

template <typename T, int IS_COMPLEX>
__aicore__ inline void RealV2Op<T, IS_COMPLEX>::Compute(int64_t currentNum)
{
    if constexpr (IS_COMPLEX == 0) {
        // Real passthrough: nothing to compute. Data already in passBuf.
        return;
    } else {
        // Complex extraction: use Gather to extract real parts from even indices
        AscendC::LocalTensor<T> inLocal = inputQueue.template DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outputQueue.template AllocTensor<T>();

        // offsetBuf contains byte offsets: [0, 2*sizeof(T), 4*sizeof(T), ...]
        AscendC::LocalTensor<uint32_t> offsetLocal = offsetBuf.Get<uint32_t>();
        AscendC::Gather(outLocal, inLocal, offsetLocal, (uint32_t)0, static_cast<uint32_t>(currentNum));

        outputQueue.template EnQue<T>(outLocal);
        inputQueue.FreeTensor(inLocal);
    }
}

template <typename T, int IS_COMPLEX>
__aicore__ inline void RealV2Op<T, IS_COMPLEX>::CopyOut(int64_t progress, int64_t currentNum)
{
    if constexpr (IS_COMPLEX == 0) {
        // Real passthrough: copy from TBuf directly to GM
        AscendC::LocalTensor<T> local = passBuf.Get<T>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGM[progress * ubLength_], local, copyParams);
        // Wait for MTE3 to complete before next iteration reuses the buffer
        AscendC::PipeBarrier<PIPE_ALL>();
    } else {
        AscendC::LocalTensor<T> outLocal = outputQueue.template DeQue<T>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(outputGM[progress * ubLength_], outLocal, copyParams);
        outputQueue.FreeTensor(outLocal);
    }
}

template <typename T, int IS_COMPLEX>
__aicore__ inline void RealV2Op<T, IS_COMPLEX>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsRealV2
#endif // REAL_V2_H
