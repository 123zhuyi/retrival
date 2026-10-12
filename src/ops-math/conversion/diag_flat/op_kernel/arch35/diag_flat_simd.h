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
 * \file diag_flat_simd.h
 * \brief DiagFlat SIMD outer shell + SIMT kernel (arch35, DAV_3510)
 *
 * Design: DESIGN.md v2.1 Sec 3.4
 *
 * Execution flow (per tile):
 *   AllocTensor → asc_vf_call (SIMT: fill UB — diagonal=input, rest=0)
 *               → EnQue → DeQue → DataCopyPad (UB → GM, continuous burst) → FreeTensor
 *
 * Note: SIMT kernel writes to EVERY UB position (even non-diagonal), matching diag_v2's
 * unconditional-write pattern, to ensure compiler emits full-vector stores with
 * 32B-aligned addresses. Conditional "scatter" writes would produce masked partial-vector
 * stores whose start address may not be 32B-aligned → VEC_ERROR 340 (ub addr misaligned).
 *
 * (ref: diag_v2/op_kernel/arch35/diag_v2.h)
 */

#ifndef __DIAG_FLAT_ARCH35_SIMD_H__
#define __DIAG_FLAT_ARCH35_SIMD_H__

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "diag_flat_tiling_data.h"  // kept for backward compat; fields now in DiagV2Arch35TilingData

using namespace AscendC;

constexpr int32_t DIAG_FLAT_BUFFER_NUM = 2;
constexpr uint32_t DIAG_FLAT_THREAD_NUM = 1024;

// ================================================================
// SIMT kernel: fill UB buffer (diagonal ← xGm, non-diagonal ← 0)
// Every thread always writes — no divergent UB store.
// (ref: batch_to_space_nd_simt.h HALF_THREAD_NUM_LAUNCH_BOUND=1024 for arch35)
// ================================================================

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(DIAG_FLAT_THREAD_NUM) void SimtDiagFlat(
    __gm__ T* xGm,          // input (1D contiguous)
    int64_t tileOffset,     // global output element offset of this tile
    int64_t tileElems,      // number of output elements in this tile
    int64_t outWidth,       // W = N + |k|
    uint64_t outWidthMagic, // fast div magic  for outWidth
    uint64_t outWidthShift, // fast div shift  for outWidth
    int64_t diagonal,       // k (signed)
    int64_t numInput,       // N = total input elements
    __ubuf__ T* yUb)        // UB output buffer (packed) — LAST
{
    for (int64_t pos = threadIdx.x; pos < tileElems; pos += blockDim.x) {
        int64_t globalPos = tileOffset + pos;

        // Fast unsigned division replaces slow int64_t / and %
        // (ref: batch_to_space_nd_simt.h Simt::UintDiv pattern)
        uint64_t row = Simt::UintDiv(static_cast<uint64_t>(globalPos), outWidthMagic, outWidthShift);
        uint64_t col = static_cast<uint64_t>(globalPos) - row * static_cast<uint64_t>(outWidth);

        int64_t inputIdx = (diagonal >= 0) ? static_cast<int64_t>(row) : static_cast<int64_t>(col);
        bool onDiag = (static_cast<int64_t>(col) - static_cast<int64_t>(row) == diagonal);
        bool inRange = (inputIdx >= 0) && (inputIdx < numInput);
        bool doRead = onDiag && inRange;

        // Always read from GM (non-divergent), then branchless select
        T gmVal = xGm[doRead ? inputIdx : 0];
        yUb[pos] = doRead ? gmVal : static_cast<T>(0);
    }
}

// ================================================================
// SIMD outer shell: pipe/buffer management + SIMT launch + DMA output
// ================================================================

template <typename T>
class DiagFlatSimd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 const DiagFlatArch35TilingData* tilingData)
    {
        td_ = tilingData;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        uint64_t blockIdx = GetBlockIdx();
        tileOffset_ = blockIdx * td_->outPerCore;
        remaining_ = min(td_->outPerCore, td_->outTotal - tileOffset_);

        // InitBuffer 3rd arg = per-buffer bytes (ref: diag_v2, batch_to_space_nd).
        uint32_t bufBytes = static_cast<uint32_t>(
            ((td_->tileLength + DIAG_FLAT_THREAD_NUM - 1) / DIAG_FLAT_THREAD_NUM) *
            DIAG_FLAT_THREAD_NUM * sizeof(T));
        pipe_.InitBuffer(outQueue_, DIAG_FLAT_BUFFER_NUM, bufBytes);

        // Pre-compute fast division magic for outWidth (ref: batch_to_space_nd pattern)
        GetUintDivMagicAndShift(outWidthMagic_, outWidthShift_,
                                static_cast<uint64_t>(td_->outWidth));
    }

    __aicore__ inline void Process()
    {
        if (remaining_ <= 0) return;

        int64_t end = tileOffset_ + remaining_;
        int64_t curTileLen;
        for (int64_t curStart = tileOffset_; curStart < end; curStart += td_->tileLength) {
            curTileLen = min(td_->tileLength, end - curStart);
            // 1. Allocate UB buffer
            auto outBuf = outQueue_.AllocTensor<T>();
            auto yUb = reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr());


            // 2. SIMT: fill UB (diagonal ← input, non-diagonal ← 0)
            // Always use fixed threadNum, same as batch_to_space_nd pattern.
            // Loop condition `pos < tileElems` naturally filters idle threads.
            asc_vf_call<SimtDiagFlat<T>>(
                dim3(DIAG_FLAT_THREAD_NUM),
                (__gm__ T*)xGm_.GetPhyAddr(),
                curStart,
                curTileLen,
                td_->outWidth,
                outWidthMagic_,
                outWidthShift_,
                td_->diagonal,
                td_->numInput,
                yUb);

            // 3. EnQue → DeQue (sync)
            outQueue_.EnQue(outBuf);
            LocalTensor<T> readyBuf = outQueue_.DeQue<T>();

            // 4. DMA: continuous burst output
            DataCopyPad(yGm_[curStart], readyBuf,
                {1, static_cast<uint32_t>(curTileLen * sizeof(T)), 0, 0, 0});

            // 5. Free UB buffer
            outQueue_.FreeTensor(outBuf);
        }
    }

private:
    const DiagFlatArch35TilingData* td_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DIAG_FLAT_BUFFER_NUM> outQueue_;
    int64_t tileOffset_;
    int64_t remaining_;
    uint64_t outWidthMagic_;
    uint64_t outWidthShift_;
};

#endif // __DIAG_FLAT_ARCH35_SIMD_H__
