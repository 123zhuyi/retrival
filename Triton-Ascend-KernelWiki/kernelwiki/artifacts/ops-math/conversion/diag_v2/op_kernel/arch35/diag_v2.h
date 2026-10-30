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
 * \file diag_v2.h
 * \brief DiagV2 SIMD outer shell + SIMT kernel (arch35, DAV_3510)
 *
 * Design: DESIGN.md v2.3 Sec 3.4
 *
 * Execution flow (per tile):
 *   AllocTensor → asc_vf_call (SIMT: GM gather → UB write)
 *               → EnQue → DeQue → DataCopyPad (UB → GM) → FreeTensor
 *
 * (ref: tile_with_axis.h for SIMD class structure;
 *       batch_to_space_nd_simt.h for __simt_vf__ pattern)
 */

#ifndef __DIAG_V2_ARCH35_H__
#define __DIAG_V2_ARCH35_H__

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "diag_v2_tiling_data.h"

using namespace AscendC;

constexpr int32_t DIAG_V2_BUFFER_NUM = 2;

// ================================================================
// SIMT kernel: diagonal element gather from GM → UB
// ================================================================

template <typename T>
__simt_vf__ __aicore__ void SimtDiagV2(
    __gm__ T* xGm,          // input matrix in GM
    __ubuf__ T* yUb,        // output buffer in UB
    int64_t curTileStart,   // start index (global output index)
    int64_t curTileLen,     // number of elements in this tile
    int64_t xWidth,         // input matrix width N
    int64_t diagonal)       // diagonal offset k
{
    for (int64_t idx = threadIdx.x; idx < curTileLen; idx += blockDim.x) {
        int64_t globalIdx = curTileStart + idx;

        // Compute 2D (row, col) from 1D output index
        int64_t row, col;
        if (diagonal >= 0) {
            row = globalIdx;
            col = globalIdx + diagonal;
        } else {
            row = globalIdx - diagonal;
            col = globalIdx;
        }

        yUb[idx] = xGm[row * xWidth + col];
    }
}

// ================================================================
// SIMD outer shell: pipe/buffer management + SIMT launch + DMA output
// ================================================================

template <typename T>
class DiagV2Simd {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 const DiagV2Arch35TilingData* tilingData)
    {
        td_ = tilingData;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);
        pipe_.InitBuffer(outQueue_, DIAG_V2_BUFFER_NUM,
                         static_cast<uint32_t>(td_->tileLength * sizeof(T)));

        uint64_t blockIdx = GetBlockIdx();
        startIdx_ = blockIdx * td_->numPerCore;
        endIdx_   = min(startIdx_ + td_->numPerCore, td_->numOut);
    }

    __aicore__ inline void Process()
    {
        if (startIdx_ >= endIdx_) return;

        int64_t curTileStart = startIdx_;

        while (curTileStart < endIdx_) {
            int64_t curTileLen = min(td_->tileLength, endIdx_ - curTileStart);

            // 1. Allocate UB buffer
            auto outBuf = outQueue_.AllocTensor<T>();
            __ubuf__ T* yUb = reinterpret_cast<__ubuf__ T*>(outBuf.GetPhyAddr());

            // 2. Launch SIMT kernel: gather diagonal elements GM → UB
            asc_vf_call<SimtDiagV2<T>>(
                dim3(static_cast<uint32_t>(curTileLen)),
                (__gm__ T*)xGm_.GetPhyAddr(),
                yUb,
                curTileStart,
                curTileLen,
                td_->xWidth,
                td_->diagonal);

            // 3. EnQue: mark buffer ready for consumer
            outQueue_.EnQue(outBuf);

            // 4. DeQue: get ready buffer for DMA read
            LocalTensor<T> readyBuf = outQueue_.DeQue<T>();

            // 5. DMA: Copy UB → GM output
            DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen   = static_cast<uint16_t>(curTileLen * sizeof(T));
            copyParams.srcStride  = 0;
            copyParams.dstStride  = 0;
            DataCopyPad(yGm_[curTileStart], readyBuf, copyParams);

            // 6. Free UB buffer
            outQueue_.FreeTensor(outBuf);

            curTileStart += curTileLen;
        }
    }

private:
    const DiagV2Arch35TilingData* td_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DIAG_V2_BUFFER_NUM> outQueue_;
    int64_t startIdx_;
    int64_t endIdx_;
};

#endif // __DIAG_V2_ARCH35_H__
