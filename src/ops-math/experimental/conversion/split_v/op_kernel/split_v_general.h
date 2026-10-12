/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_GENERAL_H_
#define OP_KERNEL_SPLIT_V_GENERAL_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_tiling_data.h"
#include "split_v_kernel_common.h"

using namespace AscendC;

template <typename T>
class SplitVGeneral : private SplitVTileCopyState<T> {
    SPLIT_V_DECLARE_TILE_COPY_API(SplitVGeneral);

private:
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    using Base = SplitVTileCopyState<T>;
    SPLIT_V_USE_TILE_COPY_STATE(Base);

    uint64_t midTileTotal;
};

template <typename T>
__aicore__ inline void SplitVGeneral<T>::Init(GM_ADDR input, GM_ADDR output, SplitVTilingData* tiling, TPipe* pipe)
{
    blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    LoadTileCopyTiling(output, tiling);
    midTileTotal = midLength * tileNum;
    LoadTileCopyLoop(tiling);
    InitTileCopyBuffer(input, outerLength * midLength * innerLength, pipe);
}

template <typename T>
__aicore__ inline void SplitVGeneral<T>::Process()
{
    const uint64_t firstTaskIdx = loopOffset;
    const uint64_t firstOuterIdx = firstTaskIdx / midTileTotal;
    const uint64_t firstMidIdx = (firstTaskIdx % midTileTotal) / tileNum;

    int64_t splitIdx = 0;
    uint64_t splitStart = 0;
    uint64_t splitLength = static_cast<uint64_t>(sizeSplits[splitIdx]);
    while (firstMidIdx >= splitStart + splitLength) {
        splitStart += splitLength;
        ++splitIdx;
        splitLength = static_cast<uint64_t>(sizeSplits[splitIdx]);
    }

    __gm__ T* dstPtr = outList.template GetDataPtr<__gm__ T>(splitIdx);
    outputGm.SetGlobalBuffer(dstPtr, outerLength * splitLength * innerLength);

    uint64_t prevOuterIdx = firstOuterIdx;

    for (uint64_t i = 0; i < loopNum; ++i) {
        const uint64_t taskIdx = loopOffset + i;
        const uint64_t outerIdx = taskIdx / midTileTotal;
        const uint64_t midIdx = (taskIdx % midTileTotal) / tileNum;
        const uint32_t tileIdx = static_cast<uint32_t>(taskIdx % tileNum);

        if (outerIdx != prevOuterIdx) {
            splitStart = 0;
            splitIdx = 0;
            splitLength = static_cast<uint64_t>(sizeSplits[0]);
            dstPtr = outList.template GetDataPtr<__gm__ T>(0);
            outputGm.SetGlobalBuffer(dstPtr, outerLength * splitLength * innerLength);
            prevOuterIdx = outerIdx;
        }

        while (midIdx >= splitStart + splitLength) {
            splitStart += splitLength;
            ++splitIdx;
            splitLength = static_cast<uint64_t>(sizeSplits[splitIdx]);
            dstPtr = outList.template GetDataPtr<__gm__ T>(splitIdx);
            outputGm.SetGlobalBuffer(dstPtr, outerLength * splitLength * innerLength);
        }

        const uint64_t srcOff = outerIdx * midLength * innerLength + midIdx * innerLength +
                                static_cast<uint64_t>(tileIdx) * tileLength;
        const uint64_t dstOff = outerIdx * splitLength * innerLength + (midIdx - splitStart) * innerLength +
                                static_cast<uint64_t>(tileIdx) * tileLength;
        const uint32_t length = (tileIdx == tileNum - 1) ? lastTileLength : tileLength;
        const uint32_t alignedLength = (length + alignedNum - 1) / alignedNum * alignedNum;
        CopyIn(srcOff, length, alignedLength);
        CopyOut(dstOff, length, alignedLength);
    }
}

template <typename T>
__aicore__ inline void SplitVGeneral<T>::CopyIn(uint64_t offset, uint32_t length, uint32_t alignedLength)
{
    SplitVQueueCopyIn(queue, inputGm, offset, length, alignedLength);
}

template <typename T>
__aicore__ inline void SplitVGeneral<T>::CopyOut(uint64_t offset, uint32_t length, uint32_t alignedLength)
{
    SplitVQueueCopyOut(queue, outputGm, offset, length, alignedLength);
}

#endif // OP_KERNEL_SPLIT_V_GENERAL_H_
