/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_LARGE_OUTER_H_
#define OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_LARGE_OUTER_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVSameLenCompactLargeOuter : private SplitVCompactState<T, 1> {
public:
    __aicore__ inline SplitVSameLenCompactLargeOuter() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataSameLenCompact* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t TRANS_BLOCK = 16;
    static constexpr uint32_t SUB_OUTER_TILE = 256;
    static constexpr uint32_t OUTER_TILE_FIXED = 2048;

    using Base = SplitVCompactState<T, BUFFER_NUM>;
    SPLIT_V_USE_COMPACT_STATE(Base);

private:
    __aicore__ inline uint32_t GetSplitSize(uint32_t splitIdx) const;

    template <typename VT>
    __aicore__ inline void ProcessTileTyped(uint64_t outerBase, uint32_t outerReal);
};

template <typename T>
__aicore__ inline void SplitVSameLenCompactLargeOuter<T>::Init(GM_ADDR input, GM_ADDR output,
                                                               SplitVTilingDataSameLenCompact* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    LoadSameLenCompactTiling(output, tiling, blockIdx, TRANS_BLOCK);
    inputGm.SetGlobalBuffer((__gm__ T*)input, totalLength);

    const uint32_t inputElems = outerTile * rowLength;
    const uint32_t transElems = SUB_OUTER_TILE * rowPitch;
    const uint32_t splitElems = SUB_OUTER_TILE * splitSize;
    InitSameLenCompactBuffers(pipe, BUFFER_NUM, inputElems, transElems, splitElems);
}

template <typename T>
__aicore__ inline uint32_t SplitVSameLenCompactLargeOuter<T>::GetSplitSize(uint32_t splitIdx) const
{
    return (splitIdx + 1 == splitNum) ? tailSplitSize : splitSize;
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompactLargeOuter<T>::ProcessTileTyped(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inReadyRaw = CopyInSameLenCompactTile(outerBase, outerReal);
    SplitVSync<HardEvent::MTE2_V>();
    LocalTensor<VT> inBits = inReadyRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> transBits = transTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> segBits = segTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> outBits = inReadyRaw.template ReinterpretCast<VT>();

    for (uint32_t subBase = 0; subBase < outerReal; subBase += SUB_OUTER_TILE) {
        const uint32_t remainRows = outerReal - subBase;
        const uint32_t subReal = remainRows > SUB_OUTER_TILE ? SUB_OUTER_TILE : remainRows;
        SplitVFirstVnchw<VT, TRANS_BLOCK>(inBits[subBase * rowLength], transBits, rowLength);

        for (uint32_t g = 0; g < splitNum; ++g) {
            const uint32_t curSplitSize = GetSplitSize(g);
            const uint32_t outputElems = subReal * curSplitSize;
            const uint32_t outputCopyElems = this->template TransposeSameLenCompactSegment<VT, TRANS_BLOCK>(
                segBits, transBits, outBits, g, curSplitSize, outputElems);

            outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(g),
                                     outerLength * static_cast<uint64_t>(curSplitSize));
            const uint64_t outputOffset = (outerBase + static_cast<uint64_t>(subBase)) *
                                          static_cast<uint64_t>(curSplitSize);
            CopyOutSameLenCompactSegment(inReadyRaw, outputOffset, outputElems, outputCopyElems);
            SplitVSync<HardEvent::MTE3_V>();
        }
    }
    queue.FreeTensor(inReadyRaw);
}

template <typename T>
__aicore__ inline void SplitVSameLenCompactLargeOuter<T>::Process()
{
    if (outerTile != OUTER_TILE_FIXED || rowLength == 0 || splitSize == 0 || splitNum == 0 || rowPitch != rowLength ||
        colChunkNum != 1 || chunkSplitNum != splitNum) {
        return;
    }

    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t tileIdx = loopOff + i;
        const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
        const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            ProcessTileTyped<half>(outerBase, outerReal);
        } else if constexpr (AscendC::Std::is_same<T, uint16_t>::value) {
            ProcessTileTyped<int16_t>(outerBase, outerReal);
        } else {
            ProcessTileTyped<T>(outerBase, outerReal);
        }
    }
}

#endif // OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_LARGE_OUTER_H_
