/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_COPY_H
#define CAST_COPY_H

#include "cast_base.h"

namespace AscendC {

class CastCopy {
public:
    __aicore__ inline CastCopy() {}
    __aicore__ inline CastCopy(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const CastTilingData& tiling)
    {
        InitParams(tiling);
        SetGmAddr(x, y);
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        int32_t loops = batchSize / ubProcessNum;
        int32_t tail = batchSize % ubProcessNum;
        for (int i = 0; i < loops; i++) {
            CopyIn(ubProcessNum, i * ubProcessNum);
            CopyOut(ubProcessNum, i * ubProcessNum);
        }
        if (tail > 0) {
            CopyIn(tail, loops * ubProcessNum);
            CopyOut(tail, loops * ubProcessNum);
        }
    }

private:
    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 0> queBind;
    GlobalTensor<uint8_t> xGm;
    GlobalTensor<uint8_t> yGm;
    LocalTensor<uint8_t> local;
    int32_t batchSize;
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

    __aicore__ inline void InitBuffers() { pipe.InitBuffer(queBind, BUFFER_NUM, ubProcessNum * sizeof(uint8_t)); }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR y)
    {
        xGm.SetGlobalBuffer((__gm__ uint8_t*)x + globalOffset);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + globalOffset);
    }

    __aicore__ inline void CopyIn(int32_t length, uint64_t offset)
    {
        queBind.AllocTensor<uint8_t>(local);
        DataCopy(local, xGm[offset], AlignUp(length, BLOCK_SIZE / sizeof(uint8_t)));
        queBind.EnQue<uint8_t>(local);
    }

    __aicore__ inline void CopyOut(int32_t length, uint64_t offset)
    {
        queBind.DeQue<uint8_t>(local);
        if (length >= BLOCK_SIZE / sizeof(uint8_t)) {
            DataCopy(yGm[offset], local, length);
        }
        int32_t alignedLen = AlignUp(length, BLOCK_SIZE / sizeof(uint8_t));
        if (alignedLen > length) {
            PipeBarrier<PIPE_ALL>();
            for (int i = alignedLen - BLOCK_SIZE / sizeof(uint8_t); i < length; i++) {
                yGm.SetValue(offset + i, local(i));
            }
        }
        queBind.FreeTensor<uint8_t>(local);
    }
};

} // namespace AscendC
#endif // CAST_COPY_H
