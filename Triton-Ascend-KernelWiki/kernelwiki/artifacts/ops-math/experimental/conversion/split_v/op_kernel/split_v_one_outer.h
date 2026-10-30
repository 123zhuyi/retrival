/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_ONE_OUTER_H_
#define OP_KERNEL_SPLIT_V_ONE_OUTER_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_tiling_data.h"
#include "split_v_kernel_common.h"

using namespace AscendC;

template <typename T>
class SplitV : private SplitVTileCopyState<T> {
    SPLIT_V_DECLARE_TILE_COPY_API(SplitV);

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    using Base = SplitVTileCopyState<T>;
    SPLIT_V_USE_TILE_COPY_STATE(Base);
};

template <typename T>
__aicore__ inline void SplitV<T>::Init(GM_ADDR input, GM_ADDR output, SplitVTilingData* tiling, TPipe* pipe)
{
    blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    LoadTileCopyTiling(output, tiling);
    LoadTileCopyLoop(tiling);
    InitTileCopyBuffer(input, midLength * innerLength, pipe);
}

template <typename T>
__aicore__ inline void SplitV<T>::Process()
{
    const uint64_t firstMidIdx = loopOffset / tileNum;
    int64_t splitIdx = 0;
    uint64_t splitStart = 0;
    uint64_t splitLength = static_cast<uint64_t>(sizeSplits[splitIdx]);
    while (firstMidIdx >= splitStart + splitLength) {
        splitStart += splitLength;
        ++splitIdx;
        splitLength = static_cast<uint64_t>(sizeSplits[splitIdx]);
    }

    __gm__ T* dstPtr = outList.template GetDataPtr<__gm__ T>(splitIdx);
    outputGm.SetGlobalBuffer(dstPtr, splitLength * innerLength);

    for (uint64_t i = 0; i < loopNum; ++i) {
        const uint64_t taskIdx = loopOffset + i;
        const uint64_t midIdx = taskIdx / tileNum;
        const uint32_t tileIdx = taskIdx % tileNum;

        while (midIdx >= splitStart + splitLength) {
            splitStart += splitLength;
            ++splitIdx;
            splitLength = static_cast<uint64_t>(sizeSplits[splitIdx]);
            dstPtr = outList.template GetDataPtr<__gm__ T>(splitIdx);
            outputGm.SetGlobalBuffer(dstPtr, splitLength * innerLength);
        }

        const uint64_t srcOff = midIdx * innerLength + static_cast<uint64_t>(tileIdx) * tileLength;
        const uint64_t dstOff = (midIdx - splitStart) * innerLength + static_cast<uint64_t>(tileIdx) * tileLength;
        const uint32_t length = (tileIdx == tileNum - 1) ? lastTileLength : tileLength;
        const uint32_t alignedLength = (length + alignedNum - 1) / alignedNum * alignedNum;
        CopyIn(srcOff, length, alignedLength);
        CopyOut(dstOff, length, alignedLength);
    }
}

template <typename T>
__aicore__ inline void SplitV<T>::CopyIn(uint64_t offset, uint32_t length, uint32_t alignedLength)
{
    SplitVQueueCopyIn(queue, inputGm, offset, length, alignedLength);
}

template <typename T>
__aicore__ inline void SplitV<T>::CopyOut(uint64_t offset, uint32_t length, uint32_t alignedLength)
{
    SplitVQueueCopyOut(queue, outputGm, offset, length, alignedLength);
}

#endif // OP_KERNEL_SPLIT_V_ONE_OUTER_H_
