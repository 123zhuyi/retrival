/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_grad_common.h
 * \brief CdistGrad arch22 CRTP base — TQue double-buffered pipeline + two-phase deterministic reduce
 *
 * All inputs arrive as broadcast [B, P, Q, M] continuous tensors (aclnn UnsqueezeNd + BroadcastTo).
 * Kernel computes gradX1[b,i,:] = sum_j grad[b,i,j] * f(x1[b,i,:], x2[b,j,:], cdist[b,i,j])
 * fully vectorized: grad/cdist are M-wide vectors (scalar repeated along k by broadcast).
 *
 * Derived classes (CdistGradP0/P1/P2/PInf/PGeneral) implement ComputeBatch(base, rows),
 * which evaluates the per-j term for `rows` consecutive j rows at once, and optionally
 * PrepareChunk(currentRTile) / AccumulateBatch(base, rows).
 *
 * Batched evaluation is the whole point of the layout: every operation in the term is
 * elementwise, so one Sub/Mul/Select over the [rows, mAligned] block replaces `rows`
 * single-repeat instructions. On a typical shape (M = 64 -> mAligned = 64) the per-row form
 * issued one 256-byte vector op per instruction and was bound by instruction issue, not by
 * the vector pipe. Only the j-reduce stays per row, so the accumulation order -- and with it
 * the rounding of the result -- is exactly what the per-row form produced.
 *
 * Synchronization (AscendC pipeline model):
 *   - Intra-pipe (e.g. Sub -> Mul -> Div -> Add in ComputeBatch): same-pipe FIFO, no sync needed.
 *   - Inter-pipe MTE2->V (chunk CopyIn): TQue<VECIN> EnQue/DeQue.
 *   - Inter-pipe V->MTE3 (CopyOut): TQue<VECOUT> EnQue/DeQue.
 *   - Inter-pipe MTE2->V on a TBuf (x1Row load, Phase2 ws read): SetFlag/WaitFlag<MTE2_V>.
 *
 * Determinism: Q-split path uses two-phase workspace reduce (Phase1 partial sums to workspace
 * slot per sub-task, SyncAll, Phase2 merge in fixed qPart ascending order), NOT atomics.
 */

#ifndef CDIST_GRAD_COMMON_H
#define CDIST_GRAD_COMMON_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../cdist_grad_tiling_data_arch22.h"
#include "../cdist_grad_tiling_key_arch22.h"

namespace NsCdistGrad {

using namespace AscendC;

constexpr int64_t BLOCK_BYTES = 32;

__aicore__ inline uint32_t AlignUpBlock(uint32_t bytes)
{
    return ((bytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
}

template <typename T, typename Derived>
class CdistGradBase {
public:
    static constexpr bool IS_FP16 = std::is_same_v<T, half>;

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult, GM_ADDR gradX1,
                                GM_ADDR workspace, const CdistGradTilingData* tilingData);
    __aicore__ inline void Process();

protected:
    static constexpr int64_t ROW_ALIGN = BLOCK_BYTES / static_cast<int64_t>(sizeof(T));
    static constexpr int64_t CMP_ALIGN = 64;
    static __aicore__ inline int64_t AlignUpSeg(int64_t v) { return ((v + ROW_ALIGN - 1) / ROW_ALIGN) * ROW_ALIGN; }
    static __aicore__ inline uint32_t CmpCount(int64_t n)
    {
        return static_cast<uint32_t>(((n + CMP_ALIGN - 1) / CMP_ALIGN) * CMP_ALIGN);
    }

    // Data views for the derived ComputeBatch(): fp32 rows of the current chunk.
    __aicore__ inline void ComputeChunk(int64_t currentRTile);
    // Default j-reduce: add each computed term row into the accumulator, in ascending j.
    __aicore__ inline void AccumulateBatch(int64_t base, int64_t rows);
    __aicore__ inline void ProcessSubTask(int64_t subIdx);
    // P-blocked task: pTile_ consecutive output rows in one pass. See PBlockRows in the tiling.
    __aicore__ inline void ProcessBlockedTask(int64_t taskIdx);
    __aicore__ inline void CopyInBlocked(int64_t rowStart, int64_t slabRows);
    __aicore__ inline void CopyOutBlocked(int64_t rowStart, int64_t rows);
    // diff = x1 - x2 for `rows` batch rows. On the blocked path x1 is a full slab and this is
    // one instruction; otherwise it is the per-row broadcast of the single x1 row.
    __aicore__ inline void SubX1(const LocalTensor<float>& dst, int64_t off, int64_t rows, uint32_t count);
    __aicore__ inline void CopyInX1Row(int64_t b, int64_t i, int64_t mStart, int64_t mTileReal);
    __aicore__ inline void CopyInChunk(int64_t b, int64_t i, int64_t rStart, int64_t currentRTile, int64_t mStart,
                                       int64_t mTileReal);
    __aicore__ inline void ResetAccumCompensation() {}
    __aicore__ inline void FoldAccumCompensation() {}

    __aicore__ inline void CopyOutPartial(int64_t mTileReal); // Phase1: accum -> ws slot segment
    __aicore__ inline void CopyOutAccum(int64_t mTileReal);   // direct write gradX1 segment
    __aicore__ inline void MergeToGradX1();                   // Phase2: fixed-order merge workspace -> gradX1
    __aicore__ inline void CopyOutRowToGradX1(const LocalTensor<float>& accum, int64_t row, int64_t mStart,
                                              int64_t mTileReal);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> x2Queue;
    TQue<QuePosition::VECIN, 1> x1Queue;
    TQue<QuePosition::VECIN, 1> gradQueue;
    TQue<QuePosition::VECIN, 1> distQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 1> rowInQueue;

    // Fixed row-wide compute buffers (fp32, mAligned_ elements).
    TBuf<QuePosition::VECCALC> x1RowBuf;
    TBuf<QuePosition::VECCALC> accumBuf;
    TBuf<QuePosition::VECCALC> maskBuf;   // batch-level Compare bit map
    TBuf<QuePosition::VECCALC> maskBuf2;  // second batch-level Compare bit map
    TBuf<QuePosition::VECCALC> tmpBuf;    // p-general high-precision scratch (9 batch rows)
    TBuf<QuePosition::VECCALC> wsReadBuf; // reserved fp32 row

    TBuf<QuePosition::VECCALC> termBuf;
    TBuf<QuePosition::VECCALC> sc1Buf;
    TBuf<QuePosition::VECCALC> sc2Buf;
    TBuf<QuePosition::VECCALC> sc3Buf;

    // fp16 cast targets for the chunk inputs. fp32 reads the queue tensors directly.
    TBuf<QuePosition::VECCALC> x1CalcBuf; // blocked + fp16 only
    TBuf<QuePosition::VECCALC> x2CalcBuf;
    TBuf<QuePosition::VECCALC> gradCalcBuf;
    TBuf<QuePosition::VECCALC> distCalcBuf;

    // Constants, batch-wide so Compare/Select can run over a whole [rows, mAligned] block.
    TBuf<QuePosition::VECCALC> zeroBuf;
    TBuf<QuePosition::VECCALC> oneBuf;
    TBuf<QuePosition::VECCALC> negOneBuf;

    GlobalTensor<T> gradOutputGM;
    GlobalTensor<T> x1GM;
    GlobalTensor<T> x2GM;
    GlobalTensor<T> cdistResultGM;
    GlobalTensor<T> gradX1GM;
    GlobalTensor<float> wsGM; // two-phase workspace (always fp32)

    // Tiling parameters.
    int64_t batchSize_ = 0;
    int64_t pSize_ = 0;
    int64_t rSize_ = 0;
    int64_t mSize_ = 0;
    int64_t mAligned_ = 0;     // CURRENT segment aligned width (all row buffers & vector counts)
    int64_t mAlignedFull_ = 0; // full-row aligned width (workspace slot stride only)
    int64_t mTileSize_ = 0;
    int64_t numMTiles_ = 1;
    int64_t lastMTileSize_ = 0;
    int64_t mStart_ = 0; // current segment start offset within the row
    int64_t rTile_ = 0;
    int64_t cTile_ = 1;
    int64_t pTile_ = 1;
    int64_t numRChunks_ = 0;
    int64_t lastRChunkSize_ = 0;
    int64_t tasksPerCore_ = 0;
    int64_t tailCoreTasks_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t qSplit_ = 1;
    int64_t qPartSize_ = 0;
    int64_t tmpBufSize_ = 0;
    float pValueF_ = 2.0f;

    // Runtime state.
    int64_t startSubTask_ = 0;
    int64_t subTaskCount_ = 0;
    bool useTwoPhase_ = false;
    int64_t currentRow_ = 0;    // (b*P+i) for current sub-task
    int64_t currentRTile_ = 0;  // rows in the chunk being computed (debug/derived use)
    int64_t currentSubIdx_ = 0; // global sub-task index
    // fp32 views of the current chunk (set in ComputeChunk).
    LocalTensor<float> x2Chunk_;
    LocalTensor<float> gradChunk_;
    LocalTensor<float> distChunk_;
    LocalTensor<float> x1Row_;
    LocalTensor<float> x1Chunk_; // blocked path only: [pTile*Q, mAligned] view of x1
    LocalTensor<float> accum_;
    LocalTensor<float> term_;
    LocalTensor<float> sc1_;
    LocalTensor<float> sc2_;
    LocalTensor<float> sc3_;
    LocalTensor<float> zero_;
    LocalTensor<float> one_;
    LocalTensor<float> negOne_;
};

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::Init(GM_ADDR gradOutput, GM_ADDR x1, GM_ADDR x2, GM_ADDR cdistResult,
                                                       GM_ADDR gradX1, GM_ADDR workspace,
                                                       const CdistGradTilingData* tilingData)
{
    batchSize_ = tilingData->batchSize;
    pSize_ = tilingData->pSize;
    rSize_ = tilingData->rSize;
    mSize_ = tilingData->mSize;
    mAlignedFull_ = tilingData->mAligned;
    mTileSize_ = tilingData->mTileSize;
    numMTiles_ = tilingData->numMTiles;
    lastMTileSize_ = tilingData->lastMTileSize;
    mAligned_ = AlignUpSeg(mTileSize_); // widest segment (runtime re-set per segment)
    rTile_ = tilingData->rTile;
    cTile_ = tilingData->cTile > 0 ? tilingData->cTile : tilingData->rTile;
    pTile_ = tilingData->pTile > 0 ? tilingData->pTile : 1;
    numRChunks_ = tilingData->numRChunks;
    lastRChunkSize_ = tilingData->lastRChunkSize;
    tasksPerCore_ = tilingData->tasksPerCore;
    tailCoreTasks_ = tilingData->tailCoreTasks;
    usedCoreNum_ = tilingData->usedCoreNum;
    qSplit_ = tilingData->qSplit;
    qPartSize_ = tilingData->qPartSize;
    tmpBufSize_ = tilingData->tmpBufSize;
    pValueF_ = tilingData->pValueF;

    int64_t blockIdx = AscendC::GetBlockIdx();
    startSubTask_ = blockIdx * tasksPerCore_;
    if (blockIdx < usedCoreNum_ - 1) {
        subTaskCount_ = tasksPerCore_;
    } else {
        subTaskCount_ = tailCoreTasks_;
    }
    if (blockIdx >= usedCoreNum_) {
        subTaskCount_ = 0;
    }
    useTwoPhase_ = (qSplit_ > 1);

    // GM tensors. All inputs are broadcast [B,P,Q,M]; output gradX1 is [B,P,M].
    int64_t pqm = batchSize_ * pSize_ * rSize_ * mSize_;
    gradOutputGM.SetGlobalBuffer((__gm__ T*)gradOutput, pqm);
    x1GM.SetGlobalBuffer((__gm__ T*)x1, pqm);
    x2GM.SetGlobalBuffer((__gm__ T*)x2, pqm);
    cdistResultGM.SetGlobalBuffer((__gm__ T*)cdistResult, pqm);
    gradX1GM.SetGlobalBuffer((__gm__ T*)gradX1, batchSize_ * pSize_ * mSize_);
    if (useTwoPhase_) {
        AscendC::SetSysWorkspace(workspace); // required before GetUserWorkspace
        GM_ADDR ws = AscendC::GetUserWorkspace(workspace);
        int64_t totalSubTasks = batchSize_ * pSize_ * qSplit_;
        wsGM.SetGlobalBuffer((__gm__ float*)ws, totalSubTasks * mAlignedFull_);
    }

    int64_t chunkElems = rTile_ * mAligned_;
    int64_t chunkBytes = (chunkElems + CMP_ALIGN) * static_cast<int64_t>(sizeof(T));
    pipe.InitBuffer(x2Queue, 2, chunkBytes);
    if (pTile_ > 1) {
        pipe.InitBuffer(x1Queue, 2, chunkBytes);
    }
    pipe.InitBuffer(gradQueue, 2, chunkBytes);
    pipe.InitBuffer(distQueue, 2, chunkBytes);
    // Output queue (single buffered). Sized fp32-wide — it carries the fp32 workspace partial
    // in Phase1 as well as the T-typed gradX1 row. The blocked path writes pTile rows at once.
    pipe.InitBuffer(outQueue, 1, pTile_ * mAligned_ * static_cast<int64_t>(sizeof(float)));

    // Fixed row-wide buffers.
    int64_t mBytes = mAligned_ * static_cast<int64_t>(sizeof(float));
    // Batch buffers carry CMP_ALIGN elements of slack: a batch-wide Compare/Select rounds its
    // count up to a 256B multiple and may touch that many elements past the last row.
    int64_t batchElems = chunkElems + CMP_ALIGN;
    int64_t chunkFp32Bytes = batchElems * static_cast<int64_t>(sizeof(float));
    pipe.InitBuffer(x1RowBuf, mBytes);
    // The blocked path keeps one accumulator per output row in the block.
    pipe.InitBuffer(accumBuf, pTile_ * mBytes);
    // Compare writes one bit per element, so a batch-wide compare needs a batch-wide bitmap.
    int64_t chunkMaskBytes = batchElems / 8;
    if (chunkMaskBytes < 32)
        chunkMaskBytes = 32;
    pipe.InitBuffer(maskBuf, chunkMaskBytes);
    pipe.InitBuffer(maskBuf2, chunkMaskBytes);
    if (tmpBufSize_ > 0) {
        pipe.InitBuffer(tmpBuf, tmpBufSize_);
    } else {
        pipe.InitBuffer(tmpBuf, 32);
    }
    // pgeneral carries its compensated-accumulation residual here, one row per accumulator.
    pipe.InitBuffer(wsReadBuf, pTile_ * mBytes);

    // Batch-wide scratch and constants.
    pipe.InitBuffer(termBuf, chunkFp32Bytes);
    pipe.InitBuffer(sc1Buf, chunkFp32Bytes);
    pipe.InitBuffer(sc2Buf, chunkFp32Bytes);
    pipe.InitBuffer(sc3Buf, chunkFp32Bytes);
    pipe.InitBuffer(zeroBuf, chunkFp32Bytes);
    pipe.InitBuffer(oneBuf, chunkFp32Bytes);
    pipe.InitBuffer(negOneBuf, chunkFp32Bytes);

    if constexpr (IS_FP16) {
        // fp32 compute views of the fp16 chunk inputs. fp32 reads the queue tensors directly.
        pipe.InitBuffer(x2CalcBuf, chunkFp32Bytes);
        pipe.InitBuffer(gradCalcBuf, chunkFp32Bytes);
        pipe.InitBuffer(distCalcBuf, chunkFp32Bytes);
        if (pTile_ > 1) {
            pipe.InitBuffer(x1CalcBuf, chunkFp32Bytes);
        }
    }

    term_ = termBuf.Get<float>();
    sc1_ = sc1Buf.Get<float>();
    sc2_ = sc2Buf.Get<float>();
    sc3_ = sc3Buf.Get<float>();
    // Constants (fp32), filled once over the widest batch so any prefix is valid.
    zero_ = zeroBuf.Get<float>();
    one_ = oneBuf.Get<float>();
    negOne_ = negOneBuf.Get<float>();
    Duplicate(zero_, 0.0f, static_cast<uint32_t>(batchElems));
    Duplicate(one_, 1.0f, static_cast<uint32_t>(batchElems));
    Duplicate(negOne_, -1.0f, static_cast<uint32_t>(batchElems));
    accum_ = accumBuf.Get<float>();

    // MTE2->V event (shared by x1Row load and Phase2 workspace read).
    // fp32-wide: reused in Phase2 to read an fp32 workspace row.
    pipe.InitBuffer(rowInQueue, 1, mAligned_ * static_cast<int64_t>(sizeof(float)));
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyInX1Row(int64_t b, int64_t i, int64_t mStart, int64_t mTileReal)
{
    // x1[b,i,0,mStart:mEnd] — j-direction broadcast row, this M segment.
    int64_t gmOffset = (b * pSize_ + i) * rSize_ * mSize_ + mStart;
    LocalTensor<T> x1In = rowInQueue.AllocTensor<T>();
    DataCopyPad(x1In, x1GM[gmOffset], {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(T))), 0, 0},
                {false, 0, 0, 0});
    rowInQueue.EnQue(x1In);
    LocalTensor<T> x1Ready = rowInQueue.DeQue<T>();
    // Copy out of the queue slot into the dedicated TBuf (both dtypes): aliasing the
    // queue tensor past FreeTensor races the next MTE2 fill of the reused slot.
    LocalTensor<float> x1Row = x1RowBuf.Get<float>();
    if constexpr (IS_FP16) {
        Cast(x1Row, x1Ready, RoundMode::CAST_NONE, static_cast<uint32_t>(mAligned_));
    } else {
        Adds(x1Row, x1Ready, 0.0f, static_cast<uint32_t>(mAligned_));
    }
    rowInQueue.FreeTensor(x1Ready);
    x1Row_ = x1Row;
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyInChunk(int64_t b, int64_t i, int64_t rStart,
                                                              int64_t currentRTile, int64_t mStart, int64_t mTileReal)
{
    const uint32_t rowLen = static_cast<uint32_t>(mTileReal * static_cast<int64_t>(sizeof(T)));
    const uint32_t srcGap = static_cast<uint32_t>((mSize_ - mTileReal) * static_cast<int64_t>(sizeof(T)));
    const uint32_t dstGap = static_cast<uint32_t>((mAligned_ * static_cast<int64_t>(sizeof(T)) - AlignUpBlock(rowLen)) /
                                                  BLOCK_BYTES);
    const DataCopyExtParams params{static_cast<uint16_t>(currentRTile), rowLen, srcGap, dstGap, 0};
    const DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // x2[b, i=0, rStart:rEnd, :] — x2 repeated along i.
    int64_t x2Base = b * pSize_ * rSize_ * mSize_ + rStart * mSize_ + mStart;
    LocalTensor<T> x2 = x2Queue.AllocTensor<T>();
    DataCopyPad(x2, x2GM[x2Base], params, padParams);
    x2Queue.EnQue(x2);

    // grad[b, i, rStart:rEnd, :] — contiguous, scalar grad[b,i,j] repeated along k.
    int64_t gradBase = ((b * pSize_ + i) * rSize_ + rStart) * mSize_ + mStart;
    LocalTensor<T> grad = gradQueue.AllocTensor<T>();
    DataCopyPad(grad, gradOutputGM[gradBase], params, padParams);
    gradQueue.EnQue(grad);

    // cdist[b, i, rStart:rEnd, :] — contiguous, scalar dist[b,i,j] repeated along k.
    LocalTensor<T> dist = distQueue.AllocTensor<T>();
    DataCopyPad(dist, cdistResultGM[gradBase], params, padParams);
    distQueue.EnQue(dist);
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::ComputeChunk(int64_t currentRTile)
{
    LocalTensor<T> x2 = x2Queue.DeQue<T>();
    LocalTensor<T> grad = gradQueue.DeQue<T>();
    LocalTensor<T> dist = distQueue.DeQue<T>();
    // The blocked path carries x1 as a fourth chunk instead of a single broadcast row.
    LocalTensor<T> x1;
    if (pTile_ > 1) {
        x1 = x1Queue.template DeQue<T>();
    }

    if constexpr (IS_FP16) {
        uint32_t chunkCount = static_cast<uint32_t>(currentRTile * mAligned_);
        LocalTensor<float> x2f = x2CalcBuf.Get<float>();
        LocalTensor<float> gradf = gradCalcBuf.Get<float>();
        LocalTensor<float> distf = distCalcBuf.Get<float>();
        Cast(x2f, x2, RoundMode::CAST_NONE, chunkCount);
        Cast(gradf, grad, RoundMode::CAST_NONE, chunkCount);
        Cast(distf, dist, RoundMode::CAST_NONE, chunkCount);
        x2Chunk_ = x2f;
        gradChunk_ = gradf;
        distChunk_ = distf;
        // The casts are the only readers of the slots, so they can be released immediately.
        x2Queue.FreeTensor(x2);
        gradQueue.FreeTensor(grad);
        distQueue.FreeTensor(dist);
        if (pTile_ > 1) {
            LocalTensor<float> x1f = x1CalcBuf.template Get<float>();
            Cast(x1f, x1, RoundMode::CAST_NONE, chunkCount);
            x1Chunk_ = x1f;
            x1Queue.FreeTensor(x1);
        }
    } else {
        x2Chunk_ = x2.template ReinterpretCast<float>();
        gradChunk_ = grad.template ReinterpretCast<float>();
        distChunk_ = dist.template ReinterpretCast<float>();
        if (pTile_ > 1) {
            x1Chunk_ = x1.template ReinterpretCast<float>();
        }
    }

    static_cast<Derived*>(this)->PrepareChunk(currentRTile);

    // Evaluate the term for cTile_ j rows at a time, then fold that batch into the
    // accumulator one row at a time (ascending j) so the reduce keeps its original order.
    for (int64_t base = 0; base < currentRTile; base += cTile_) {
        int64_t rows = currentRTile - base;
        if (rows > cTile_) {
            rows = cTile_;
        }
        static_cast<Derived*>(this)->ComputeBatch(base, rows);
        static_cast<Derived*>(this)->AccumulateBatch(base, rows);
    }

    if constexpr (!IS_FP16) {
        x2Queue.FreeTensor(x2);
        gradQueue.FreeTensor(grad);
        distQueue.FreeTensor(dist);
        if (pTile_ > 1) {
            x1Queue.FreeTensor(x1);
        }
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::AccumulateBatch(int64_t base, int64_t rows)
{
    if (pTile_ > 1) {
        const int64_t pOut = rows / rSize_;
        const uint8_t rowBlocks = static_cast<uint8_t>(mAligned_ / (BLOCK_BYTES / sizeof(float)));
        const AscendC::BinaryRepeatParams params(1, 1, 1, rowBlocks, rowBlocks,
                                                 static_cast<uint8_t>(rSize_ * rowBlocks));
        for (int64_t j = 0; j < rSize_; j++) {
            AscendC::Add(accum_, accum_, term_[(base + j) * mAligned_], static_cast<uint64_t>(mAligned_),
                         static_cast<uint8_t>(pOut), params);
        }
        return;
    }
    uint32_t count = static_cast<uint32_t>(mAligned_);
    for (int64_t k = 0; k < rows; k++) {
        AscendC::Add(accum_, accum_, term_[(base + k) * mAligned_], count);
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::SubX1(const LocalTensor<float>& dst, int64_t off, int64_t rows,
                                                        uint32_t count)
{
    if (pTile_ > 1) {
        AscendC::Sub(dst, x1Chunk_[off], x2Chunk_[off], count);
        return;
    }
    for (int64_t k = 0; k < rows; k++) {
        AscendC::Sub(dst[k * mAligned_], x1Row_, x2Chunk_[off + k * mAligned_], static_cast<uint32_t>(mAligned_));
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyInBlocked(int64_t rowStart, int64_t slabRows)
{
    const int64_t base = rowStart * rSize_ * mSize_;
    const uint32_t rowLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(T)));
    const uint32_t dstGap = static_cast<uint32_t>((mAligned_ * static_cast<int64_t>(sizeof(T)) - AlignUpBlock(rowLen)) /
                                                  BLOCK_BYTES);
    const bool packed = (mSize_ == mAligned_);
    const DataCopyExtParams params = packed ?
                                         DataCopyExtParams{
                                             1, static_cast<uint32_t>(slabRows * static_cast<int64_t>(rowLen)), 0, 0,
                                             0} :
                                         DataCopyExtParams{static_cast<uint16_t>(slabRows), rowLen, 0, dstGap, 0};
    const DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> x1 = x1Queue.template AllocTensor<T>();
    DataCopyPad(x1, x1GM[base], params, padParams);
    x1Queue.EnQue(x1);
    LocalTensor<T> x2 = x2Queue.template AllocTensor<T>();
    DataCopyPad(x2, x2GM[base], params, padParams);
    x2Queue.EnQue(x2);
    LocalTensor<T> grad = gradQueue.template AllocTensor<T>();
    DataCopyPad(grad, gradOutputGM[base], params, padParams);
    gradQueue.EnQue(grad);
    LocalTensor<T> dist = distQueue.template AllocTensor<T>();
    DataCopyPad(dist, cdistResultGM[base], params, padParams);
    distQueue.EnQue(dist);
}

// accum holds [rows, mAligned]; the destination region of gradX1 is [rows, M], contiguous.
template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutBlocked(int64_t rowStart, int64_t rows)
{
    const int64_t gmOffset = rowStart * mSize_;
    const uint32_t count = static_cast<uint32_t>(rows * mAligned_);
    const uint32_t rowLen = static_cast<uint32_t>(mSize_ * static_cast<int64_t>(sizeof(T)));
    // UB rows are mAligned_ elements apart while the copy reads AlignUpBlock(rowLen) bytes of
    // each, so the source gap is the remainder; the destination is contiguous.
    const uint32_t srcGap = static_cast<uint32_t>((mAligned_ * static_cast<int64_t>(sizeof(T)) - AlignUpBlock(rowLen)) /
                                                  BLOCK_BYTES);
    // Same single-descriptor collapse as CopyInBlocked when the rows need no padding.
    const bool packed = (mSize_ == mAligned_);
    const DataCopyExtParams params = packed ?
                                         DataCopyExtParams{
                                             1, static_cast<uint32_t>(rows * static_cast<int64_t>(rowLen)), 0, 0, 0} :
                                         DataCopyExtParams{static_cast<uint16_t>(rows), rowLen, srcGap, 0, 0};
    if constexpr (IS_FP16) {
        LocalTensor<half> outT = outQueue.template AllocTensor<half>();
        Cast(outT, accum_, RoundMode::CAST_ROUND, count);
        outQueue.EnQue(outT);
        LocalTensor<half> outY = outQueue.template DeQue<half>();
        DataCopyPad(gradX1GM[gmOffset], outY, params);
        outQueue.FreeTensor(outY);
    } else {
        LocalTensor<float> outT = outQueue.template AllocTensor<float>();
        Adds(outT, accum_, 0.0f, count);
        outQueue.EnQue(outT);
        LocalTensor<float> outY = outQueue.template DeQue<float>();
        DataCopyPad(gradX1GM[gmOffset], outY, params);
        outQueue.FreeTensor(outY);
    }
}

// One task = pTile_ consecutive flattened (b,i) output rows. Reachable only when the host
// enabled blocking, which requires numMTiles == 1 and qSplit == 1, so there is no M-segment
// loop and no two-phase reduce here.
template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::ProcessBlockedTask(int64_t taskIdx)
{
    const int64_t totalRows = batchSize_ * pSize_;
    const int64_t rowStart = taskIdx * pTile_;
    if (rowStart >= totalRows) {
        return;
    }
    int64_t rows = totalRows - rowStart;
    if (rows > pTile_) {
        rows = pTile_;
    }
    const int64_t slabRows = rows * rSize_;
    currentRow_ = rowStart;
    currentRTile_ = slabRows;

    CopyInBlocked(rowStart, slabRows);
    Duplicate(accum_, 0.0f, static_cast<uint32_t>(rows * mAligned_));
    static_cast<Derived*>(this)->ResetAccumCompensation();
    ComputeChunk(slabRows);
    static_cast<Derived*>(this)->FoldAccumCompensation();
    CopyOutBlocked(rowStart, rows);
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::ProcessSubTask(int64_t subIdx)
{
    int64_t taskIdx = subIdx / qSplit_;
    int64_t qPart = subIdx % qSplit_;
    int64_t b = taskIdx / pSize_;
    int64_t i = taskIdx % pSize_;
    currentRow_ = taskIdx;
    currentSubIdx_ = subIdx;

    int64_t qStart = qPart * qPartSize_;
    int64_t qEnd = qStart + qPartSize_;
    if (qEnd > rSize_)
        qEnd = rSize_;

    // ---- M-segment loop: each segment has its own x1 slice, accumulator, chunk pass ----
    for (int64_t mSeg = 0; mSeg < numMTiles_; mSeg++) {
        mStart_ = mSeg * mTileSize_;
        int64_t mTileReal = (mSeg == numMTiles_ - 1) ? lastMTileSize_ : mTileSize_;
        mAligned_ = AlignUpSeg(mTileReal);

        CopyInX1Row(b, i, mStart_, mTileReal);

        Duplicate(accum_, 0.0f, static_cast<uint32_t>(mAligned_));
        static_cast<Derived*>(this)->ResetAccumCompensation();

        for (int64_t chunk = 0; chunk < numRChunks_; chunk++) {
            int64_t chunkStart = chunk * rTile_;
            int64_t chunkSize = (chunk == numRChunks_ - 1) ? lastRChunkSize_ : rTile_;
            int64_t rStart = (chunkStart > qStart) ? chunkStart : qStart;
            int64_t rEnd = (chunkStart + chunkSize < qEnd) ? chunkStart + chunkSize : qEnd;
            if (rStart >= rEnd)
                continue;
            int64_t currentRTile = rEnd - rStart;

            CopyInChunk(b, i, rStart, currentRTile, mStart_, mTileReal);
            ComputeChunk(currentRTile);
        }

        static_cast<Derived*>(this)->FoldAccumCompensation();

        if (useTwoPhase_) {
            CopyOutPartial(mTileReal);
        } else {
            CopyOutRowToGradX1(accum_, currentRow_, mStart_, mTileReal);
        }
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutRowToGradX1(const LocalTensor<float>& accum, int64_t row,
                                                                     int64_t mStart, int64_t mTileReal)
{
    int64_t gmOffset = row * mSize_ + mStart;
    if constexpr (IS_FP16) {
        LocalTensor<half> outT = outQueue.AllocTensor<half>();
        Cast(outT, accum, RoundMode::CAST_ROUND, static_cast<uint32_t>(mAligned_));
        outQueue.EnQue(outT);
        LocalTensor<half> outY = outQueue.DeQue<half>();
        DataCopyPad(gradX1GM[gmOffset], outY,
                    {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(half))), 0, 0});
        outQueue.FreeTensor(outY);
    } else {
        LocalTensor<float> outT = outQueue.AllocTensor<float>();
        Adds(outT, accum, 0.0f, static_cast<uint32_t>(mAligned_));
        outQueue.EnQue(outT);
        LocalTensor<float> outY = outQueue.DeQue<float>();
        DataCopyPad(gradX1GM[gmOffset], outY,
                    {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(float))), 0, 0});
        outQueue.FreeTensor(outY);
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutAccum(int64_t mTileReal)
{
    CopyOutRowToGradX1(accum_, currentRow_, mStart_, mTileReal);
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::CopyOutPartial(int64_t mTileReal)
{
    int64_t wsOffset = currentSubIdx_ * mAlignedFull_ + mStart_;
    // Partial sums stay fp32 for BOTH dtypes — see the wsGM declaration.
    LocalTensor<float> outT = outQueue.AllocTensor<float>();
    Adds(outT, accum_, 0.0f, static_cast<uint32_t>(mAligned_));
    outQueue.EnQue(outT);
    LocalTensor<float> outY = outQueue.DeQue<float>();
    DataCopyPad(wsGM[wsOffset], outY,
                {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(float))), 0, 0});
    outQueue.FreeTensor(outY);
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::MergeToGradX1()
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalRows = batchSize_ * pSize_;
    int64_t rowsPerCore = (totalRows + usedCoreNum_ - 1) / usedCoreNum_;
    int64_t rowStart = blockIdx * rowsPerCore;
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > totalRows)
        rowEnd = totalRows;
    if (rowStart >= rowEnd)
        return;

    LocalTensor<float> partial = sc1_;
    for (int64_t row = rowStart; row < rowEnd; row++) {
        for (int64_t mSeg = 0; mSeg < numMTiles_; mSeg++) {
            mStart_ = mSeg * mTileSize_;
            int64_t mTileReal = (mSeg == numMTiles_ - 1) ? lastMTileSize_ : mTileSize_;
            mAligned_ = AlignUpSeg(mTileReal);
            Duplicate(accum_, 0.0f, static_cast<uint32_t>(mAligned_));
            // Fixed ascending qPart order -> bit-identical deterministic result.
            for (int64_t q = 0; q < qSplit_; q++) {
                int64_t wsOffset = (row * qSplit_ + q) * mAlignedFull_ + mStart_;
                LocalTensor<float> wsIn = rowInQueue.AllocTensor<float>();
                DataCopyPad(wsIn, wsGM[wsOffset],
                            {1, static_cast<uint16_t>(mTileReal * static_cast<int64_t>(sizeof(float))), 0, 0},
                            {false, 0, 0, 0});
                rowInQueue.EnQue(wsIn);
                LocalTensor<float> wsReady = rowInQueue.DeQue<float>();
                Adds(partial, wsReady, 0.0f, static_cast<uint32_t>(mAligned_));
                Add(accum_, accum_, partial, static_cast<uint32_t>(mAligned_));
                rowInQueue.FreeTensor(wsReady);
            }
            CopyOutRowToGradX1(accum_, row, mStart_, mTileReal);
        }
    }
}

template <typename T, typename Derived>
__aicore__ inline void CdistGradBase<T, Derived>::Process()
{
    if (useTwoPhase_) {
        for (int64_t t = 0; t < subTaskCount_; t++) {
            ProcessSubTask(startSubTask_ + t);
        }
        // Drain MTE3 before the cross-core barrier: SyncAll only fences instruction
        // flow arrival, workspace writes (MTE3) may still be in flight on other cores.
        PipeBarrier<PIPE_ALL>();
        SyncAll(); // ensure all partial sums written
        MergeToGradX1();
        PipeBarrier<PIPE_ALL>(); // drain MTE3 (gradX1 writes) before final barrier
        SyncAll();               // ensure all gradX1 rows written
    } else if (pTile_ > 1) {
        for (int64_t t = 0; t < subTaskCount_; t++) {
            ProcessBlockedTask(startSubTask_ + t);
        }
    } else {
        for (int64_t t = 0; t < subTaskCount_; t++) {
            ProcessSubTask(startSubTask_ + t); // segment-wise CopyOut inside
        }
    }
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_COMMON_H
