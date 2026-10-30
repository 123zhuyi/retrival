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
 * \file complex.h
 * \brief ComplexV3 operator kernel class definition (arch32)
 *
 * Template parameters:
 *   - T: Data type (float / half)
 *   - BROADCAST_MODE: 0=no broadcast (same shape), 1=needs broadcast
 *
 * No-broadcast mode (BROADCAST_MODE=0):
 *   Uses double-buffered pipeline: CopyIn -> Interleave -> CopyOut
 *   real[i] and imag[i] are interleaved into out[2*i] and out[2*i+1]
 *
 * Broadcast mode (BROADCAST_MODE=1):
 *   Uses element-wise GM access with broadcast index mapping.
 *   Reads real/imag from GM using broadcast stride-based index computation,
 *   interleaves into outBuf, and writes to GM.
 *   Only outQueue is used (no realQueue/imagQueue).
 */
#ifndef COMPLEX_V3_H
#define COMPLEX_V3_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "complex_v3_tiling_data.h"
#include "complex_v3_tiling_key.h"

namespace NsComplexV3 {

using namespace AscendC;

template <typename T, int BROADCAST_MODE>
class ComplexV3 {
    static constexpr int32_t BUFFER_NUM = 2; // Double buffer

public:
    __aicore__ inline ComplexV3(){};

    __aicore__ inline void Init(GM_ADDR real, GM_ADDR imag, GM_ADDR out, const ComplexV3TilingData* tilingData);
    __aicore__ inline void Process();

private:
    // No-broadcast path methods
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Interleave(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

    // Broadcast path methods
    __aicore__ inline void ProcessBroadcastPreload();
    __aicore__ inline void ProcessBroadcastOnDemand();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> realQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> imagQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<T> realGM;
    GlobalTensor<T> imagGM;
    GlobalTensor<T> outGM;

    // Broadcast mode: UB copies of full real/imag inputs (preloadMode=1)
    TBuf<QuePosition::VECCALC> realTBuf;
    TBuf<QuePosition::VECCALC> imagTBuf;
    LocalTensor<T> realLocal_;
    LocalTensor<T> imagLocal_;

    // On-demand mode: temporary UB buffers for per-chunk loading (preloadMode=0)
    TBuf<QuePosition::VECCALC> realTmpBuf;
    TBuf<QuePosition::VECCALC> imagTmpBuf;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    const ComplexV3TilingData* tilingData_ = nullptr;
};

// =============================================================================
// Init
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::Init(
    GM_ADDR real, GM_ADDR imag, GM_ADDR out, const ComplexV3TilingData* tilingData)
{
    tilingData_ = tilingData;
    int64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
    int64_t remainder = tilingData->totalLength - offset;
    blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
    ubLength_ = tilingData->ubFactor;

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        blockLength_ = 0;
        return;
    }

    if constexpr (BROADCAST_MODE == 0) {
        realGM.SetGlobalBuffer((__gm__ T*)real + offset, blockLength_);
        imagGM.SetGlobalBuffer((__gm__ T*)imag + offset, blockLength_);
        outGM.SetGlobalBuffer((__gm__ T*)out + offset * 2, blockLength_ * 2);

        pipe.InitBuffer(realQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(imagQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubLength_ * 2 * sizeof(T));
    } else {
        int64_t realSize = tilingData->realInputSize;
        int64_t imagSize = tilingData->imagInputSize;
        realGM.SetGlobalBuffer((__gm__ T*)real, realSize);
        imagGM.SetGlobalBuffer((__gm__ T*)imag, imagSize);
        outGM.SetGlobalBuffer((__gm__ T*)out + offset * 2, blockLength_ * 2);
        pipe.InitBuffer(outQueue, BUFFER_NUM, ubLength_ * 2 * sizeof(T));

        if (tilingData->preloadMode == 1) {
            // Full preload: allocate UB buffers for entire real/imag inputs
            int64_t realBufBytes = ((realSize * static_cast<int64_t>(sizeof(T)) + 31) / 32) * 32;
            int64_t imagBufBytes = ((imagSize * static_cast<int64_t>(sizeof(T)) + 31) / 32) * 32;
            pipe.InitBuffer(realTBuf, realBufBytes);
            pipe.InitBuffer(imagTBuf, imagBufBytes);
            realLocal_ = realTBuf.Get<T>();
            imagLocal_ = imagTBuf.Get<T>();
        } else {
            // On-demand: allocate temp buffers sized to ubFactor for per-chunk loading
            int64_t tmpBufBytes = ((ubLength_ * static_cast<int64_t>(sizeof(T)) + 31) / 32) * 32;
            pipe.InitBuffer(realTmpBuf, tmpBufBytes);
            pipe.InitBuffer(imagTmpBuf, tmpBufBytes);
        }
    }
}

// =============================================================================
// CopyIn - No broadcast: copy contiguous data from GM to UB
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> realLocal = realQueue.template AllocTensor<T>();
    LocalTensor<T> imagLocal = imagQueue.template AllocTensor<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(realLocal, realGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    DataCopyPad(imagLocal, imagGM[progress * ubLength_], copyParams, {false, 0, 0, 0});

    realQueue.EnQue(realLocal);
    imagQueue.EnQue(imagLocal);
}

// =============================================================================
// Interleave - Pack (real, imag) pairs using wider-type reinterpret writes
//
//   fp32: ReinterpretCast<uint64_t>, each pair packed as r | (im << 32)
//   fp16: ReinterpretCast<uint32_t>, each pair packed as r | (im << 16)
//   4x loop unrolling for throughput
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::Interleave(int64_t currentNum)
{
    LocalTensor<T> realLocal = realQueue.template DeQue<T>();
    LocalTensor<T> imagLocal = imagQueue.template DeQue<T>();
    LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

    if constexpr (sizeof(T) == 4) {
        // fp32: pack (real, imag) as uint64_t
        LocalTensor<uint64_t> outWide = outLocal.template ReinterpretCast<uint64_t>();
        LocalTensor<uint32_t> realU32 = realLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> imagU32 = imagLocal.template ReinterpretCast<uint32_t>();

        int64_t i = 0;
        int64_t limit4 = currentNum - 3;
        for (; i < limit4; i += 4) {
            uint64_t r0 = realU32.GetValue(i);
            uint64_t r1 = realU32.GetValue(i + 1);
            uint64_t r2 = realU32.GetValue(i + 2);
            uint64_t r3 = realU32.GetValue(i + 3);
            uint64_t i0 = imagU32.GetValue(i);
            uint64_t i1 = imagU32.GetValue(i + 1);
            uint64_t i2 = imagU32.GetValue(i + 2);
            uint64_t i3 = imagU32.GetValue(i + 3);
            outWide.SetValue(i, r0 | (i0 << 32));
            outWide.SetValue(i + 1, r1 | (i1 << 32));
            outWide.SetValue(i + 2, r2 | (i2 << 32));
            outWide.SetValue(i + 3, r3 | (i3 << 32));
        }
        for (; i < currentNum; i++) {
            uint64_t r = realU32.GetValue(i);
            uint64_t im = imagU32.GetValue(i);
            outWide.SetValue(i, r | (im << 32));
        }
    } else {
        // fp16: pack (real, imag) as uint32_t
        LocalTensor<uint32_t> outWide = outLocal.template ReinterpretCast<uint32_t>();
        LocalTensor<uint16_t> realU16 = realLocal.template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> imagU16 = imagLocal.template ReinterpretCast<uint16_t>();

        int64_t i = 0;
        int64_t limit4 = currentNum - 3;
        for (; i < limit4; i += 4) {
            uint32_t r0 = realU16.GetValue(i);
            uint32_t r1 = realU16.GetValue(i + 1);
            uint32_t r2 = realU16.GetValue(i + 2);
            uint32_t r3 = realU16.GetValue(i + 3);
            uint32_t i0 = imagU16.GetValue(i);
            uint32_t i1 = imagU16.GetValue(i + 1);
            uint32_t i2 = imagU16.GetValue(i + 2);
            uint32_t i3 = imagU16.GetValue(i + 3);
            outWide.SetValue(i, r0 | (i0 << 16));
            outWide.SetValue(i + 1, r1 | (i1 << 16));
            outWide.SetValue(i + 2, r2 | (i2 << 16));
            outWide.SetValue(i + 3, r3 | (i3 << 16));
        }
        for (; i < currentNum; i++) {
            uint32_t r = realU16.GetValue(i);
            uint32_t im = imagU16.GetValue(i);
            outWide.SetValue(i, r | (im << 16));
        }
    }

    outQueue.template EnQue<T>(outLocal);
    realQueue.FreeTensor(realLocal);
    imagQueue.FreeTensor(imagLocal);
}

// =============================================================================
// CopyOut - Copy interleaved data from UB to GM
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outLocal = outQueue.template DeQue<T>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * 2 * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(outGM[progress * ubLength_ * 2], outLocal, copyParams);

    outQueue.FreeTensor(outLocal);
}

// =============================================================================
// ProcessBroadcastPreload - Full preload mode: all inputs fit in UB
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::ProcessBroadcastPreload()
{
    // Bulk copy real and imag inputs from GM to UB (one-time preload)
    // Note: DataCopyPad blockLen is limited to 65535 bytes per transfer.
    // For large inputs, split into multiple transfers.
    constexpr uint32_t MAX_BLOCK_LEN = 65504; // Safe limit (32-byte aligned, < 65535)
    int64_t realSize = tilingData_->realInputSize;
    int64_t imagSize = tilingData_->imagInputSize;

    // Copy real input (may need multiple transfers)
    {
        int64_t totalBytes = realSize * static_cast<int64_t>(sizeof(T));
        int64_t copied = 0;
        while (copied < totalBytes) {
            int64_t chunkBytes = totalBytes - copied;
            if (chunkBytes > MAX_BLOCK_LEN)
                chunkBytes = MAX_BLOCK_LEN;
            DataCopyParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(chunkBytes);
            p.srcStride = 0;
            p.dstStride = 0;
            int64_t elemOffset = copied / static_cast<int64_t>(sizeof(T));
            DataCopyPad(realLocal_[elemOffset], realGM[elemOffset], p, {false, 0, 0, 0});
            copied += chunkBytes;
        }
    }

    // Copy imag input (may need multiple transfers)
    {
        int64_t totalBytes = imagSize * static_cast<int64_t>(sizeof(T));
        int64_t copied = 0;
        while (copied < totalBytes) {
            int64_t chunkBytes = totalBytes - copied;
            if (chunkBytes > MAX_BLOCK_LEN)
                chunkBytes = MAX_BLOCK_LEN;
            DataCopyParams p;
            p.blockCount = 1;
            p.blockLen = static_cast<uint32_t>(chunkBytes);
            p.srcStride = 0;
            p.dstStride = 0;
            int64_t elemOffset = copied / static_cast<int64_t>(sizeof(T));
            DataCopyPad(imagLocal_[elemOffset], imagGM[elemOffset], p, {false, 0, 0, 0});
            copied += chunkBytes;
        }
    }

    // Wait for DMA transfers to complete before reading from UB
    PipeBarrier<PIPE_ALL>();

    int64_t startIdx = tilingData_->blockFactor * AscendC::GetBlockIdx();
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    for (int64_t loop = 0; loop < loopCount; loop++) {
        int64_t currentNum = (loop == loopCount - 1) ? (blockLength_ - ubLength_ * loop) : ubLength_;

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        for (int64_t i = 0; i < currentNum; i++) {
            int64_t globalIdx = startIdx + loop * ubLength_ + i;

            int64_t realIdx = 0, imagIdx = 0;
            int64_t remaining = globalIdx;
            for (int d = 0; d < static_cast<int>(tilingData_->dimNum); d++) {
                int64_t dimStride = 1;
                for (int dd = d + 1; dd < static_cast<int>(tilingData_->dimNum); dd++) {
                    dimStride *= tilingData_->outShape[dd];
                }
                int64_t coord = remaining / dimStride;
                remaining %= dimStride;
                realIdx += coord * tilingData_->realStride[d];
                imagIdx += coord * tilingData_->imagStride[d];
            }

            outLocal.SetValue(2 * i, realLocal_.GetValue(realIdx));
            outLocal.SetValue(2 * i + 1, imagLocal_.GetValue(imagIdx));
        }

        outQueue.template EnQue<T>(outLocal);
        LocalTensor<T> outResult = outQueue.template DeQue<T>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * 2 * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outGM[loop * ubLength_ * 2], outResult, copyParams);

        outQueue.FreeTensor(outResult);
    }
}

// =============================================================================
// ProcessBroadcastOnDemand - On-demand mode: inputs too large for UB
//   For each output chunk, try to load needed input ranges into temp buffers.
//   If ranges exceed buffer capacity, fall back to element-by-element loading.
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::ProcessBroadcastOnDemand()
{
    int64_t startIdx = tilingData_->blockFactor * AscendC::GetBlockIdx();
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;

    LocalTensor<T> realTmp = realTmpBuf.Get<T>();
    LocalTensor<T> imagTmp = imagTmpBuf.Get<T>();

    // Helper lambda-like inline: compute real/imag index from global output index
    // (inlined in loop below to avoid function pointer overhead)

    for (int64_t loop = 0; loop < loopCount; loop++) {
        int64_t chunkStart = loop * ubLength_;
        int64_t currentNum = (loop == loopCount - 1) ? (blockLength_ - chunkStart) : ubLength_;

        // Phase 1: Scan to find input index ranges for this output chunk
        int64_t realMin = tilingData_->realInputSize;
        int64_t realMax = 0;
        int64_t imagMin = tilingData_->imagInputSize;
        int64_t imagMax = 0;

        for (int64_t i = 0; i < currentNum; i++) {
            int64_t globalIdx = startIdx + chunkStart + i;
            int64_t realIdx = 0, imagIdx = 0;
            int64_t remaining = globalIdx;
            for (int d = 0; d < static_cast<int>(tilingData_->dimNum); d++) {
                int64_t dimStride = 1;
                for (int dd = d + 1; dd < static_cast<int>(tilingData_->dimNum); dd++) {
                    dimStride *= tilingData_->outShape[dd];
                }
                int64_t coord = remaining / dimStride;
                remaining %= dimStride;
                realIdx += coord * tilingData_->realStride[d];
                imagIdx += coord * tilingData_->imagStride[d];
            }
            if (realIdx < realMin)
                realMin = realIdx;
            if (realIdx > realMax)
                realMax = realIdx;
            if (imagIdx < imagMin)
                imagMin = imagIdx;
            if (imagIdx > imagMax)
                imagMax = imagIdx;
        }

        int64_t realRangeLen = realMax - realMin + 1;
        int64_t imagRangeLen = imagMax - imagMin + 1;
        bool rangesFit = (realRangeLen <= ubLength_) && (imagRangeLen <= ubLength_);

        LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();

        if (rangesFit) {
            // Fast path: load contiguous ranges and index into them
            {
                DataCopyParams realParams;
                realParams.blockCount = 1;
                realParams.blockLen = static_cast<uint32_t>(realRangeLen * sizeof(T));
                realParams.srcStride = 0;
                realParams.dstStride = 0;
                DataCopyPad(realTmp, realGM[realMin], realParams, {false, 0, 0, 0});
            }
            {
                DataCopyParams imagParams;
                imagParams.blockCount = 1;
                imagParams.blockLen = static_cast<uint32_t>(imagRangeLen * sizeof(T));
                imagParams.srcStride = 0;
                imagParams.dstStride = 0;
                DataCopyPad(imagTmp, imagGM[imagMin], imagParams, {false, 0, 0, 0});
            }

            PipeBarrier<PIPE_ALL>();

            for (int64_t i = 0; i < currentNum; i++) {
                int64_t globalIdx = startIdx + chunkStart + i;
                int64_t realIdx = 0, imagIdx = 0;
                int64_t remaining = globalIdx;
                for (int d = 0; d < static_cast<int>(tilingData_->dimNum); d++) {
                    int64_t dimStride = 1;
                    for (int dd = d + 1; dd < static_cast<int>(tilingData_->dimNum); dd++) {
                        dimStride *= tilingData_->outShape[dd];
                    }
                    int64_t coord = remaining / dimStride;
                    remaining %= dimStride;
                    realIdx += coord * tilingData_->realStride[d];
                    imagIdx += coord * tilingData_->imagStride[d];
                }

                outLocal.SetValue(2 * i, realTmp.GetValue(realIdx - realMin));
                outLocal.SetValue(2 * i + 1, imagTmp.GetValue(imagIdx - imagMin));
            }
        } else {
            // Slow path: ranges exceed buffer, load element by element
            for (int64_t i = 0; i < currentNum; i++) {
                int64_t globalIdx = startIdx + chunkStart + i;
                int64_t realIdx = 0, imagIdx = 0;
                int64_t remaining = globalIdx;
                for (int d = 0; d < static_cast<int>(tilingData_->dimNum); d++) {
                    int64_t dimStride = 1;
                    for (int dd = d + 1; dd < static_cast<int>(tilingData_->dimNum); dd++) {
                        dimStride *= tilingData_->outShape[dd];
                    }
                    int64_t coord = remaining / dimStride;
                    remaining %= dimStride;
                    realIdx += coord * tilingData_->realStride[d];
                    imagIdx += coord * tilingData_->imagStride[d];
                }

                // Load single real element
                {
                    DataCopyParams p;
                    p.blockCount = 1;
                    p.blockLen = static_cast<uint32_t>(sizeof(T));
                    p.srcStride = 0;
                    p.dstStride = 0;
                    DataCopyPad(realTmp, realGM[realIdx], p, {false, 0, 0, 0});
                }
                // Load single imag element
                {
                    DataCopyParams p;
                    p.blockCount = 1;
                    p.blockLen = static_cast<uint32_t>(sizeof(T));
                    p.srcStride = 0;
                    p.dstStride = 0;
                    DataCopyPad(imagTmp, imagGM[imagIdx], p, {false, 0, 0, 0});
                }

                PipeBarrier<PIPE_ALL>();

                outLocal.SetValue(2 * i, realTmp.GetValue(0));
                outLocal.SetValue(2 * i + 1, imagTmp.GetValue(0));
            }
        }

        outQueue.template EnQue<T>(outLocal);
        LocalTensor<T> outResult = outQueue.template DeQue<T>();

        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = currentNum * 2 * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outGM[chunkStart * 2], outResult, copyParams);

        outQueue.FreeTensor(outResult);
    }
}

// =============================================================================
// Process - Main entry point
// =============================================================================
template <typename T, int BROADCAST_MODE>
__aicore__ inline void ComplexV3<T, BROADCAST_MODE>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    if constexpr (BROADCAST_MODE == 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            Interleave(currentNum);
            CopyOut(i, currentNum);
        }
    } else {
        if (tilingData_->preloadMode == 1) {
            ProcessBroadcastPreload();
        } else {
            ProcessBroadcastOnDemand();
        }
    }
}

} // namespace NsComplexV3
#endif // COMPLEX_V3_H
