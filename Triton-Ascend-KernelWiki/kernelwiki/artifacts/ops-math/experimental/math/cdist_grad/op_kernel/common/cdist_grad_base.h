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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file cdist_grad_base.h
 * \brief CdistGrad base class: Init, CopyIn/CopyOut common logic, multi-core task dispatch
 *
 * All p-value specific classes inherit from this base class and implement
 * ComputeForJ() for their specific gradient computation.
 * Supports fp32 (direct) and fp16 (mixed precision: Cast to fp32, compute, Cast back).
 */
#ifndef CDIST_GRAD_BASE_H
#define CDIST_GRAD_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_tiling_data.h"
#include "cdist_grad_tiling_key.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradBase {
public:
    __aicore__ inline CdistGradBase() {}

    __aicore__ inline void Init(
        GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult, GM_ADDR gradX1,
        const CdistGradTilingData* tilingData);
    __aicore__ inline void Process();

protected:
    // Note: Subclasses should implement their own ComputeForJ method.
    // Virtual dispatch is not supported in device code, so each p-value
    // class is self-contained rather than inheriting from this base.

    __aicore__ inline void CopyInX1Row(int64_t b, int64_t i);
    __aicore__ inline void CopyInRChunk(int64_t b, int64_t i, int64_t rStart, int64_t currentRTile);
    __aicore__ inline void CopyOutAccum(int64_t b, int64_t i);
    __aicore__ inline void ProcessTask(int64_t taskIdx);

protected:
    static constexpr bool IS_FP16 = std::is_same_v<T, half>;

    TPipe pipe;

    // Fixed buffers (persist across entire Process)
    TBuf<QuePosition::VECCALC> x1RowBuf;
    TBuf<QuePosition::VECCALC> accumBuf;
    TBuf<QuePosition::VECCALC> diffBuf;
    TBuf<QuePosition::VECCALC> localGradBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> distBroadcastBuf;
    TBuf<QuePosition::VECCALC> maskCalcBuf;
    TBuf<QuePosition::VECCALC> tmpReduceCalcBuf;

    // Cast buffer for fp16 CopyIn/CopyOut
    TBuf<QuePosition::VECCALC> castBuf;

    // R_tile chunk data buffers
    TBuf<QuePosition::VECCALC> x2ChunkBuf;
    TBuf<QuePosition::VECCALC> gradChunkBuf;
    TBuf<QuePosition::VECCALC> distChunkBuf;

    // GM tensors
    GlobalTensor<T> gradOutputGM;
    GlobalTensor<T> x1GM;
    GlobalTensor<T> x2GM;
    GlobalTensor<T> cdistResultGM;
    GlobalTensor<T> gradX1GM;

    // Tiling parameters
    int64_t batchSize_ = 0;
    int64_t pSize_ = 0;
    int64_t rSize_ = 0;
    int64_t mSize_ = 0;
    int64_t mAligned_ = 0;
    int64_t rTile_ = 0;
    int64_t numRChunks_ = 0;
    int64_t lastRChunkSize_ = 0;
    int64_t rTileAligned_ = 0;
    int64_t maskBufSize_ = 0;
    int64_t tmpReduceBufSize_ = 0;
    double pValue_ = 0.0;

    // Multi-core parameters
    int64_t startTask_ = 0;
    int64_t taskCount_ = 0;
};

template <typename T>
__aicore__ inline void CdistGradBase<T>::Init(
    GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult, GM_ADDR gradX1,
    const CdistGradTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    pSize_ = tilingData->pSize;
    rSize_ = tilingData->rSize;
    mSize_ = tilingData->mSize;
    mAligned_ = tilingData->mAligned;
    rTile_ = tilingData->rTile;
    numRChunks_ = tilingData->numRChunks;
    lastRChunkSize_ = tilingData->lastRChunkSize;
    rTileAligned_ = tilingData->rTileAligned;
    maskBufSize_ = tilingData->maskBufSize;
    tmpReduceBufSize_ = tilingData->tmpReduceBufSize;
    pValue_ = tilingData->pValue;

    // Multi-core split: compute task range for current core
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t tasksPerCore = tilingData->tasksPerCore;
    int64_t tailCoreTasks = tilingData->tailCoreTasks;
    int64_t usedCoreNum = tilingData->usedCoreNum;

    startTask_ = blockIdx * tasksPerCore;
    if (blockIdx < usedCoreNum - 1) {
        taskCount_ = tasksPerCore;
    } else {
        taskCount_ = tailCoreTasks;
    }

    // Setup GM pointers
    int64_t gradOutputSize = batchSize_ * pSize_ * rSize_;
    gradOutputGM.SetGlobalBuffer((__gm__ T*)gradOutput, gradOutputSize);

    int64_t x1Size = batchSize_ * pSize_ * mSize_;
    x1GM.SetGlobalBuffer((__gm__ T*)x1, x1Size);

    int64_t x2Size = batchSize_ * rSize_ * mSize_;
    x2GM.SetGlobalBuffer((__gm__ T*)x2, x2Size);

    int64_t cdistResultSize = batchSize_ * pSize_ * rSize_;
    cdistResultGM.SetGlobalBuffer((__gm__ T*)cdistResult, cdistResultSize);

    int64_t gradX1Size = batchSize_ * pSize_ * mSize_;
    gradX1GM.SetGlobalBuffer((__gm__ T*)gradX1, gradX1Size);

    // Initialize buffers (all compute in fp32)
    int64_t mBytes = mAligned_ * static_cast<int64_t>(sizeof(float));
    pipe.InitBuffer(x1RowBuf, mBytes);
    pipe.InitBuffer(accumBuf, mBytes);
    pipe.InitBuffer(diffBuf, mBytes);
    pipe.InitBuffer(localGradBuf, mBytes);
    pipe.InitBuffer(tmpBuf, mBytes);
    pipe.InitBuffer(distBroadcastBuf, mBytes);
    pipe.InitBuffer(maskCalcBuf, maskBufSize_);
    pipe.InitBuffer(tmpReduceCalcBuf, tmpReduceBufSize_);

    // Cast buffer for fp16
    if constexpr (IS_FP16) {
        int64_t castBytes = mAligned_ * static_cast<int64_t>(sizeof(half));
        if (castBytes < 32)
            castBytes = 32;
        pipe.InitBuffer(castBuf, castBytes);
    }

    // R_tile chunk buffers (fp32)
    pipe.InitBuffer(x2ChunkBuf, rTile_ * mAligned_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(gradChunkBuf, rTileAligned_ * static_cast<int64_t>(sizeof(float)));
    pipe.InitBuffer(distChunkBuf, rTileAligned_ * static_cast<int64_t>(sizeof(float)));
}

template <typename T>
__aicore__ inline void CdistGradBase<T>::CopyInX1Row(int64_t b, int64_t i)
{
    LocalTensor<float> x1Row = x1RowBuf.Get<float>();
    Duplicate(x1Row, 0.0f, static_cast<uint32_t>(mAligned_));
    PipeBarrier<PIPE_ALL>();

    int64_t gmOffset = b * pSize_ * mSize_ + i * mSize_;
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(T)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    if constexpr (IS_FP16) {
        LocalTensor<half> castLocal = castBuf.Get<half>();
        Duplicate(castLocal, static_cast<half>(0), static_cast<uint32_t>(mAligned_));
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(castLocal, x1GM[gmOffset], copyParams, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Cast(x1Row, castLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(mAligned_));
    } else {
        DataCopyPad(x1Row, x1GM[gmOffset], copyParams, {false, 0, 0, 0});
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CdistGradBase<T>::CopyInRChunk(int64_t b, int64_t i, int64_t rStart, int64_t currentRTile)
{
    // Load x2[b, rStart:rStart+currentRTile, :] into x2ChunkBuf
    LocalTensor<float> x2Chunk = x2ChunkBuf.Get<float>();
    Duplicate(x2Chunk, 0.0f, static_cast<uint32_t>(rTile_ * mAligned_));
    PipeBarrier<PIPE_ALL>();

    if constexpr (IS_FP16) {
        LocalTensor<half> castLocal = castBuf.Get<half>();
        for (int64_t j = 0; j < currentRTile; j++) {
            int64_t x2GmOffset = b * rSize_ * mSize_ + (rStart + j) * mSize_;
            int64_t ubOffset = j * mAligned_;
            DataCopyParams x2Params;
            x2Params.blockCount = 1;
            x2Params.blockLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(half)));
            x2Params.srcStride = 0;
            x2Params.dstStride = 0;
            Duplicate(castLocal, static_cast<half>(0), static_cast<uint32_t>(mAligned_));
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(castLocal, x2GM[x2GmOffset], x2Params, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            Cast(x2Chunk[ubOffset], castLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(mAligned_));
            PipeBarrier<PIPE_ALL>();
        }
    } else {
        for (int64_t j = 0; j < currentRTile; j++) {
            int64_t x2GmOffset = b * rSize_ * mSize_ + (rStart + j) * mSize_;
            int64_t ubOffset = j * mAligned_;
            DataCopyParams x2Params;
            x2Params.blockCount = 1;
            x2Params.blockLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(float)));
            x2Params.srcStride = 0;
            x2Params.dstStride = 0;
            DataCopyPad(x2Chunk[ubOffset], x2GM[x2GmOffset], x2Params, {false, 0, 0, 0});
        }
    }

    // Load grad_output[b, i, rStart:rStart+currentRTile] into gradChunkBuf
    LocalTensor<float> gradChunk = gradChunkBuf.Get<float>();
    Duplicate(gradChunk, 0.0f, static_cast<uint32_t>(rTileAligned_));
    PipeBarrier<PIPE_ALL>();

    int64_t gradGmBase = b * pSize_ * rSize_ + i * rSize_ + rStart;

    if constexpr (IS_FP16) {
        LocalTensor<half> castLocal = castBuf.Get<half>();
        int64_t castCapacity = mAligned_;
        int64_t remaining = currentRTile;
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        while (remaining > 0) {
            int64_t batchLen = remaining;
            if (batchLen > castCapacity)
                batchLen = castCapacity;
            Duplicate(castLocal, static_cast<half>(0), static_cast<uint32_t>(castCapacity));
            PipeBarrier<PIPE_ALL>();
            DataCopyParams gParams;
            gParams.blockCount = 1;
            gParams.blockLen = static_cast<uint32_t>(batchLen * static_cast<int64_t>(sizeof(half)));
            gParams.srcStride = 0;
            gParams.dstStride = 0;
            DataCopyPad(castLocal, gradOutputGM[gradGmBase + srcOffset], gParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            int64_t castCount = ((batchLen + 7) / 8) * 8;
            if (castCount > castCapacity)
                castCount = castCapacity;
            Cast(gradChunk[dstOffset], castLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(castCount));
            PipeBarrier<PIPE_ALL>();
            remaining -= batchLen;
            srcOffset += batchLen;
            dstOffset += batchLen;
        }
    } else {
        DataCopyParams gradParams;
        gradParams.blockCount = 1;
        gradParams.blockLen = static_cast<uint32_t>(currentRTile * static_cast<int64_t>(sizeof(float)));
        gradParams.srcStride = 0;
        gradParams.dstStride = 0;
        DataCopyPad(gradChunk, gradOutputGM[gradGmBase], gradParams, {false, 0, 0, 0});
    }

    // Load cdist_result[b, i, rStart:rStart+currentRTile] into distChunkBuf
    LocalTensor<float> distChunk = distChunkBuf.Get<float>();
    Duplicate(distChunk, 0.0f, static_cast<uint32_t>(rTileAligned_));
    PipeBarrier<PIPE_ALL>();

    int64_t distGmBase = b * pSize_ * rSize_ + i * rSize_ + rStart;

    if constexpr (IS_FP16) {
        LocalTensor<half> castLocal = castBuf.Get<half>();
        int64_t castCapacity = mAligned_;
        int64_t remaining = currentRTile;
        int64_t srcOffset = 0;
        int64_t dstOffset = 0;
        while (remaining > 0) {
            int64_t batchLen = remaining;
            if (batchLen > castCapacity)
                batchLen = castCapacity;
            Duplicate(castLocal, static_cast<half>(0), static_cast<uint32_t>(castCapacity));
            PipeBarrier<PIPE_ALL>();
            DataCopyParams dParams;
            dParams.blockCount = 1;
            dParams.blockLen = static_cast<uint32_t>(batchLen * static_cast<int64_t>(sizeof(half)));
            dParams.srcStride = 0;
            dParams.dstStride = 0;
            DataCopyPad(castLocal, cdistResultGM[distGmBase + srcOffset], dParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            int64_t castCount = ((batchLen + 7) / 8) * 8;
            if (castCount > castCapacity)
                castCount = castCapacity;
            Cast(distChunk[dstOffset], castLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(castCount));
            PipeBarrier<PIPE_ALL>();
            remaining -= batchLen;
            srcOffset += batchLen;
            dstOffset += batchLen;
        }
    } else {
        DataCopyParams distParams;
        distParams.blockCount = 1;
        distParams.blockLen = static_cast<uint32_t>(currentRTile * static_cast<int64_t>(sizeof(float)));
        distParams.srcStride = 0;
        distParams.dstStride = 0;
        DataCopyPad(distChunk, cdistResultGM[distGmBase], distParams, {false, 0, 0, 0});
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CdistGradBase<T>::CopyOutAccum(int64_t b, int64_t i)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> accum = accumBuf.Get<float>();
    int64_t gmOffset = b * pSize_ * mSize_ + i * mSize_;

    if constexpr (IS_FP16) {
        LocalTensor<half> castLocal = castBuf.Get<half>();
        Cast(castLocal, accum, RoundMode::CAST_ROUND, static_cast<uint32_t>(mAligned_));
        PipeBarrier<PIPE_ALL>();
        DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(half)));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(gradX1GM[gmOffset], castLocal, params);
    } else {
        DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(float)));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(gradX1GM[gmOffset], accum, params);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CdistGradBase<T>::ProcessTask(int64_t taskIdx)
{
    // Decompose task index to (b, i)
    int64_t b = taskIdx / pSize_;
    int64_t i = taskIdx % pSize_;

    // Step 1: Load x1 row (reused across all R chunks)
    CopyInX1Row(b, i);

    // Step 2: Initialize accumulator to zero
    LocalTensor<float> accum = accumBuf.Get<float>();
    Duplicate(accum, 0.0f, static_cast<uint32_t>(mAligned_));

    // Step 3: Loop over R chunks
    for (int64_t chunk = 0; chunk < numRChunks_; chunk++) {
        int64_t rStart = chunk * rTile_;
        int64_t currentRTile = rTile_;
        if (chunk == numRChunks_ - 1) {
            currentRTile = lastRChunkSize_;
        }

        // Load x2, grad_output, cdist_result chunks
        CopyInRChunk(b, i, rStart, currentRTile);

        // Process each j in the current R chunk
        for (int64_t j = 0; j < currentRTile; j++) {
            ComputeForJ(j, currentRTile);
        }
    }

    // Step 4: Write back accumulated gradient
    CopyOutAccum(b, i);
}

template <typename T>
__aicore__ inline void CdistGradBase<T>::Process()
{
    for (int64_t t = 0; t < taskCount_; t++) {
        ProcessTask(startTask_ + t);
    }
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_BASE_H
