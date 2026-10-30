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
#ifndef ACOS_ARCH32_H
#define ACOS_ARCH32_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "acos_v2_tiling_data.h"
#include "acos_v2_tiling_key.h"

namespace NsAcos {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T, uint32_t CALC_MODE>
class AcosV2Kernel {
public:
    __aicore__ inline AcosV2Kernel() {}

    __aicore__ inline void Init(GM_ADDR selfGm, GM_ADDR outGm, const AcosV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t loopIdx, int64_t tileLen);
    __aicore__ inline void Compute(int64_t tileLen);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t tileLen);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // Temporary buffers for fp16/bf16 cast path
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> acosBuf;

    GlobalTensor<T> selfGlobal;
    GlobalTensor<T> outGlobal;

    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

template <typename T, uint32_t CALC_MODE>
__aicore__ inline void AcosV2Kernel<T, CALC_MODE>::Init(
    GM_ADDR selfGm, GM_ADDR outGm, const AcosV2TilingData* tilingData)
{
    uint32_t coreIdx = AscendC::GetBlockIdx();
    int64_t totalLength = static_cast<int64_t>(tilingData->totalLength);
    int64_t blockFactor = static_cast<int64_t>(tilingData->blockFactor);
    tileLength_ = static_cast<int64_t>(tilingData->tileLength);

    // Calculate this core's workload
    int64_t coreOffset = coreIdx * blockFactor;
    int64_t remainderLength = totalLength - coreOffset;
    blockLength_ = (remainderLength > blockFactor) ? blockFactor : remainderLength;

    // Set GM buffer with core offset
    selfGlobal.SetGlobalBuffer((__gm__ T*)selfGm + coreOffset, blockLength_);
    outGlobal.SetGlobalBuffer((__gm__ T*)outGm + coreOffset, blockLength_);

    // Initialize double-buffer queues
    pipe.InitBuffer(inQueue, BUFFER_NUM, tileLength_ * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, tileLength_ * sizeof(T));

    // For fp16/bf16 mode, allocate temporary fp32 buffers for cast
    if constexpr (CALC_MODE != ACOS_MODE_FP32) {
        pipe.InitBuffer(castBuf, tileLength_ * sizeof(float));
        pipe.InitBuffer(acosBuf, tileLength_ * sizeof(float));
    }
}

template <typename T, uint32_t CALC_MODE>
__aicore__ inline void AcosV2Kernel<T, CALC_MODE>::Process()
{
    // Empty tensor or core with no work: skip all processing
    if (blockLength_ <= 0) {
        return;
    }

    int64_t loopCount = (blockLength_ + tileLength_ - 1) / tileLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentLen = (i == (loopCount - 1)) ? (blockLength_ - tileLength_ * i) : tileLength_;
        CopyIn(i, currentLen);
        Compute(currentLen);
        CopyOut(i, currentLen);
    }
}

template <typename T, uint32_t CALC_MODE>
__aicore__ inline void AcosV2Kernel<T, CALC_MODE>::CopyIn(int64_t loopIdx, int64_t tileLen)
{
    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(tileLen * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(inLocal, selfGlobal[loopIdx * tileLength_], copyParams, {false, 0, 0, 0});
    inQueue.EnQue(inLocal);
}

template <typename T, uint32_t CALC_MODE>
__aicore__ inline void AcosV2Kernel<T, CALC_MODE>::Compute(int64_t tileLen)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

    if constexpr (CALC_MODE == ACOS_MODE_FP32) {
        // fp32: direct Acos computation
        Acos(outLocal, inLocal, tileLen);
    } else {
        // fp16/bf16: Cast to fp32 -> Acos -> Cast back
        LocalTensor<float> castLocal = castBuf.Get<float>();
        LocalTensor<float> acosLocal = acosBuf.Get<float>();

        // Cast T -> fp32
        Cast(castLocal, inLocal, RoundMode::CAST_NONE, tileLen);
        // Acos on fp32
        Acos(acosLocal, castLocal, tileLen);
        // Cast fp32 -> T
        Cast(outLocal, acosLocal, RoundMode::CAST_ROUND, tileLen);
    }

    outQueue.EnQue<T>(outLocal);
    inQueue.FreeTensor(inLocal);
}

template <typename T, uint32_t CALC_MODE>
__aicore__ inline void AcosV2Kernel<T, CALC_MODE>::CopyOut(int64_t loopIdx, int64_t tileLen)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(tileLen * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(outGlobal[loopIdx * tileLength_], outLocal, copyParams);
    outQueue.FreeTensor(outLocal);
}

} // namespace NsAcos

#endif // ACOS_ARCH32_H
