/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_BASE_H
#define CAST_BASE_H

#include "kernel_operator.h"
#include "cast_tiling_data.h"

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_NUM = 2;

namespace AscendC {

template <typename T, typename U>
class CastBase {
public:
    __aicore__ inline CastBase() {}

protected:
    TPipe pipe;
    TQue<QuePosition::VECIN, 0> xQue;
    TQue<QuePosition::VECOUT, 0> yQue;
    GlobalTensor<T> xGm;
    GlobalTensor<U> yGm;
    LocalTensor<T> xLocal;
    LocalTensor<U> yLocal;

    int64_t batchSize;
    uint64_t blockIdx;
    uint64_t globalOffset;
    int32_t ubProcessNum;

    __aicore__ inline void InitParams(const CastTilingData& tiling)
    {
        blockIdx = GetBlockIdx();
        ubProcessNum = tiling.ubProcessNum;
        globalOffset = blockIdx * tiling.formerBatchSize;
        batchSize = (blockIdx < tiling.formerCoreNum) ? tiling.formerBatchSize : tiling.tailBatchSize;
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR y, GM_ADDR)
    {
        xGm.SetGlobalBuffer((__gm__ T*)x + globalOffset);
        yGm.SetGlobalBuffer((__gm__ U*)y + globalOffset);
    }

    __aicore__ inline void InitIoBuffers()
    {
        pipe.InitBuffer(xQue, BUFFER_NUM, ubProcessNum * sizeof(T));
        pipe.InitBuffer(yQue, BUFFER_NUM, ubProcessNum * sizeof(U));
    }

    __aicore__ inline void CopyIn(int32_t length, uint64_t offset)
    {
        xQue.AllocTensor<T>(xLocal);
        DataCopy(xLocal, xGm[offset], AlignUp(length, BLOCK_SIZE / sizeof(T)));
        xQue.EnQue<T>(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t length, uint64_t offset)
    {
        yQue.DeQue<U>(yLocal);
        if (length >= BLOCK_SIZE / sizeof(U)) {
            DataCopy(yGm[offset], yLocal, length);
        }
        int32_t alignedLen = AlignUp(length, BLOCK_SIZE / sizeof(U));
        if (alignedLen > length) {
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = alignedLen - BLOCK_SIZE / sizeof(U); i < length; i++) {
                yGm.SetValue(offset + i, yLocal(i));
            }
        }
        yQue.FreeTensor<U>(yLocal);
    }

    template <typename Derived>
    __aicore__ inline void RunProcess(Derived* self)
    {
        int64_t loops = batchSize / ubProcessNum;
        int32_t tail = static_cast<int32_t>(batchSize % ubProcessNum);
        for (int64_t i = 0; i < loops; i++) {
            CopyIn(ubProcessNum, i * ubProcessNum);
            self->Compute(ubProcessNum);
            CopyOut(ubProcessNum, i * ubProcessNum);
        }
        if (tail > 0) {
            CopyIn(tail, loops * ubProcessNum);
            self->Compute(tail);
            CopyOut(tail, loops * ubProcessNum);
        }
    }
};

} // namespace AscendC

#endif // CAST_BASE_H
