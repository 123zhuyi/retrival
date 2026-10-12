/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fill_diagonal_v2_sparse.h
 * \brief
 */
#ifndef FILL_DIAGONAL_V2_SPARSE_H
#define FILL_DIAGONAL_V2_SPARSE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace FillDiagonalV2Sparse {
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t UB_BLOCK_SIZE = 32;
constexpr uint64_t CACHE_BLOCK_SIZE = 128;

template <typename T>
class Kernel {
public:
    __aicore__ inline Kernel()
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR fill_value, const FillDiagonalV2TilingData* tilingData)
    {
        if (AscendC::GetBlockIdx() < AscendC::GetBlockNum() - 1) {
            blockLength = tilingData->blockLength;
        } else {
            blockLength = tilingData->lastBlockLength;
        }
        blockStart = tilingData->blockLength * AscendC::GetBlockIdx();
        blockEnd = blockStart + blockLength;
        // 处理完整的cache Line, 避免缓存一致性问题
        uint64_t cacheBlockLength = CACHE_BLOCK_SIZE / sizeof(T);
        blockStart -= (blockStart * sizeof(T) + reinterpret_cast<uint64_t>(x)) % CACHE_BLOCK_SIZE / sizeof(T);
        blockEnd += cacheBlockLength - 1;
        blockEnd -= (blockEnd * sizeof(T) + reinterpret_cast<uint64_t>(x)) % CACHE_BLOCK_SIZE / sizeof(T);
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), tilingData->totalLength);
        this->valGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(fill_value), 1);
        this->step = tilingData->step;
        this->end = tilingData->end;
        this->val = this->valGm.GetValue(0);
    }

    __aicore__ inline void Process()
    {
        Compute();
    }

private:
    __aicore__ inline void Compute()
    {
        blockStart = blockStart >= 0 ? blockStart : 0;
        blockEnd = blockEnd > this->end ? this->end : blockEnd;
        uint64_t diagIndex;
        if (blockStart % this->step == 0) {
            diagIndex = blockStart;
        } else {
            diagIndex = (blockStart / this->step + 1) * this->step;
        }
        while (diagIndex < this->end && diagIndex < blockEnd) {
            xGm.SetValue(diagIndex, this->val);
            diagIndex += this->step;
        }
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE>(xGm);
    }

    T val;
    uint64_t step;
    uint64_t end;
    uint64_t blockLength;
    uint64_t blockStart;
    uint64_t blockEnd;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> valGm;
};
} // namespace FillDiagonalV2Sparse
#endif