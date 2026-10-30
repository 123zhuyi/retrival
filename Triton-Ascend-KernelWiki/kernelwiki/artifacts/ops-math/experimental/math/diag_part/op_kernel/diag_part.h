/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DIAG_PART_H_
#define DIAG_PART_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "diag_part_tiling_data.h"
#include "diag_part_tiling_key.h"

namespace NsDiagPart {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// NPU vector engine minimum transfer = 32 bytes
// float/int32: 8 elements, float16: 16 elements
template <typename TYPE_X>
struct AlignSize {
    static constexpr int32_t VALUE = 8;
};

template <>
struct AlignSize<half> {
    static constexpr int32_t VALUE = 16;
};

template <typename TYPE_X>
class KernelDiagPart {
public:
    __aicore__ inline KernelDiagPart(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DiagPartTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessBatch(int64_t xGmOffset, int64_t yGmOffset, int32_t count);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> indexBuf;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_X> yGm;

    int64_t sideLength;
    int64_t realCoreNum;
    int64_t numPerCore; // sub-block size = ALIGN
    int64_t tailNum;    // remaining elements in last sub-block
};

template <typename TYPE_X>
__aicore__ inline void KernelDiagPart<TYPE_X>::Init(GM_ADDR x, GM_ADDR y, const DiagPartTilingData* tilingData)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(y));

    sideLength = tilingData->sideLength;
    realCoreNum = tilingData->realCoreNum;
    numPerCore = tilingData->numPerCore;
    tailNum = tilingData->tailNum;

    // Input buffer: numPerCore x numPerCore sub-block
    pipe.InitBuffer(xQueue, BUFFER_NUM, numPerCore * numPerCore * sizeof(TYPE_X));
    // Output buffer: numPerCore elements
    pipe.InitBuffer(yQueue, BUFFER_NUM, numPerCore * sizeof(TYPE_X));
    // Index buffer: numPerCore int32 values for Gather
    pipe.InitBuffer(indexBuf, numPerCore * sizeof(int32_t));
}

template <typename TYPE_X>
__aicore__ inline void KernelDiagPart<TYPE_X>::Process()
{
    int64_t blockIdx = GetBlockIdx();
    if (blockIdx >= realCoreNum) {
        return;
    }

    // Compute per-core block assignment
    int64_t totalBlocks = (sideLength + numPerCore - 1) / numPerCore;
    int64_t blocksPerCore = totalBlocks / realCoreNum;
    int64_t tailBlocks = totalBlocks % realCoreNum;

    int64_t myBlocks = blocksPerCore + ((blockIdx < tailBlocks) ? 1 : 0);
    if (myBlocks == 0) {
        return;
    }

    int64_t myStartBlock;
    if (blockIdx < tailBlocks) {
        myStartBlock = blockIdx * (blocksPerCore + 1);
    } else {
        myStartBlock = tailBlocks * (blocksPerCore + 1) + (blockIdx - tailBlocks) * blocksPerCore;
    }

    for (int64_t b = 0; b < myBlocks; b++) {
        int64_t blockIdx_ = myStartBlock + b;
        int64_t xGmOffset = blockIdx_ * numPerCore * (sideLength + 1);
        int64_t yGmOffset = blockIdx_ * numPerCore;

        int32_t count = static_cast<int32_t>(numPerCore);
        // Last block may have fewer elements
        if (blockIdx_ == totalBlocks - 1 && tailNum != 0) {
            count = static_cast<int32_t>(tailNum);
        }

        ProcessBatch(xGmOffset, yGmOffset, count);
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelDiagPart<TYPE_X>::ProcessBatch(int64_t xGmOffset, int64_t yGmOffset, int32_t count)
{
    // ========== CopyIn: Read count x count sub-block from diagonal ==========
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = static_cast<uint16_t>(count);
    copyInParams.blockLen = static_cast<uint32_t>(count * sizeof(TYPE_X));
    copyInParams.srcStride = static_cast<uint32_t>((sideLength - count) * sizeof(TYPE_X));
    copyInParams.dstStride = 0;

    DataCopyPadExtParams<TYPE_X> padParams = {false, 0, 0, 0};

    LocalTensor<TYPE_X> xLocal = xQueue.AllocTensor<TYPE_X>();
    DataCopyPad(xLocal, xGm[xGmOffset], copyInParams, padParams);

    SetFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    WaitFlag<HardEvent::MTE2_V>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    // ========== Gather: Extract diagonal elements from sub-block ==========
    // Ascend hardware aligns each row to 32-byte blocks in local memory,
    // so stride must use aligned row size, not raw count * sizeof(T).
    constexpr int32_t BLOCK_BYTES = 32;
    int32_t alignRowBytes =
        ((count * static_cast<int32_t>(sizeof(TYPE_X)) + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    ArithProgression<int32_t>(
        indexLocal, static_cast<int32_t>(0), alignRowBytes + static_cast<int32_t>(sizeof(TYPE_X)),
        static_cast<int32_t>(count));
    LocalTensor<uint32_t> indexU32 = indexLocal.ReinterpretCast<uint32_t>();

    LocalTensor<TYPE_X> yLocal = yQueue.AllocTensor<TYPE_X>();
    Gather(yLocal, xLocal, indexU32, 0, static_cast<uint32_t>(count));

    SetFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    WaitFlag<HardEvent::V_MTE3>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    // ========== CopyOut: Write output ==========
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = static_cast<uint32_t>(count * sizeof(TYPE_X));
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    DataCopyPad(yGm[yGmOffset], yLocal, copyOutParams);

    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
}

} // namespace NsDiagPart

#endif
