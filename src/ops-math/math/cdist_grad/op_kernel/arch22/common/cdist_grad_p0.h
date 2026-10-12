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
 * \file cdist_grad_p0.h
 * \brief CdistGrad p=0 — replicates 950 CdistGradP0Dag (output all zeros).
 *
 * p=0 (Hamming-like, non-differentiable): gradX1 is zero everywhere.
 * Overrides Process to write zeros per (row, M-segment), skipping the j loop.
 */

#ifndef CDIST_GRAD_P0_H
#define CDIST_GRAD_P0_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradP0 : public CdistGradBase<T, CdistGradP0<T>> {
public:
    using Base = CdistGradBase<T, CdistGradP0<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeForJ(int64_t j);
    __aicore__ inline void Process();
};

template <typename T>
__aicore__ inline void CdistGradP0<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile;
}

template <typename T>
__aicore__ inline void CdistGradP0<T>::ComputeForJ(int64_t j)
{
    (void)j; // unused: Process writes zeros directly
}

template <typename T>
__aicore__ inline void CdistGradP0<T>::Process()
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t totalRows = this->batchSize_ * this->pSize_;
    int64_t usedCore = (this->usedCoreNum_ > 0) ? this->usedCoreNum_ : 1;
    int64_t rowsPerCore = (totalRows + usedCore - 1) / usedCore;
    int64_t rowStart = blockIdx * rowsPerCore;
    int64_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > totalRows)
        rowEnd = totalRows;
    if (rowStart >= rowEnd)
        return;

    for (int64_t r = rowStart; r < rowEnd; r++) {
        for (int64_t mSeg = 0; mSeg < this->numMTiles_; mSeg++) {
            int64_t mStart = mSeg * this->mTileSize_;
            int64_t mTileReal = (mSeg == this->numMTiles_ - 1) ? this->lastMTileSize_ : this->mTileSize_;
            int64_t gmOffset = r * this->mSize_ + mStart;
            if constexpr (std::is_same_v<T, half>) {
                LocalTensor<half> outT = this->outQueue.template AllocTensor<half>();
                Duplicate(outT, static_cast<half>(0), static_cast<uint32_t>(mTileReal));
                this->outQueue.EnQue(outT);
                LocalTensor<half> outY = this->outQueue.template DeQue<half>();
                AscendC::DataCopyPad(this->gradX1GM[gmOffset], outY, {1, static_cast<uint16_t>(mTileReal * 2), 0, 0});
                this->outQueue.FreeTensor(outY);
            } else {
                LocalTensor<float> outT = this->outQueue.template AllocTensor<float>();
                Duplicate(outT, 0.0f, static_cast<uint32_t>(mTileReal));
                this->outQueue.EnQue(outT);
                LocalTensor<float> outY = this->outQueue.template DeQue<float>();
                AscendC::DataCopyPad(this->gradX1GM[gmOffset], outY, {1, static_cast<uint16_t>(mTileReal * 4), 0, 0});
                this->outQueue.FreeTensor(outY);
            }
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>(); // drain MTE3 zero-writes before kernel exit
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_P0_H
