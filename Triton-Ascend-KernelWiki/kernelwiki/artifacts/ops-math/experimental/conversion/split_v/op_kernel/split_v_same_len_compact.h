/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS SOFTWARE IS
 * PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 * repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_H_
#define OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVSameLenCompact : private SplitVCompactState<T, 1> {
public:
    __aicore__ inline SplitVSameLenCompact() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataSameLenCompact* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t TRANS_BLOCK = 16;
    static constexpr uint32_t OUTER_TILE_FIXED = 256;

    using Base = SplitVCompactState<T, BUFFER_NUM>;
    SPLIT_V_USE_COMPACT_STATE(Base);

private:
    __aicore__ inline uint32_t GetSplitSize(uint32_t splitIdx) const;

    template <typename VT>
    __aicore__ inline void ProcessTileTyped(uint64_t outerBase, uint32_t outerReal);
    template <typename VT>
    __aicore__ inline void ProcessChunkTaskTyped(uint64_t taskIdx);
};

template <typename T>
__aicore__ inline void SplitVSameLenCompact<T>::Init(GM_ADDR input, GM_ADDR output,
                                                     SplitVTilingDataSameLenCompact* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    LoadSameLenCompactTiling(output, tiling, blockIdx, TRANS_BLOCK);
    inputGm.SetGlobalBuffer((__gm__ T*)input + coreOuterBase * rowLength, coreOuterLength * rowLength);

    const uint32_t inputElems = outerTile * rowPitch;
    const uint32_t transElems = outerTile * rowPitch;
    const uint32_t splitElems = outerTile * splitSize;
    InitSameLenCompactBuffers(pipe, BUFFER_NUM, inputElems, transElems, splitElems);
}

template <typename T>
__aicore__ inline uint32_t SplitVSameLenCompact<T>::GetSplitSize(uint32_t splitIdx) const
{
    return (splitIdx + 1 == splitNum) ? tailSplitSize : splitSize;
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompact<T>::ProcessTileTyped(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inReadyRaw = CopyInSameLenCompactTile(outerBase, outerReal);
    SplitVSync<HardEvent::MTE2_V>();
    LocalTensor<VT> inBits = inReadyRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> transBits = transTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> segBits = segTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> outBits = inReadyRaw.template ReinterpretCast<VT>();

    SplitVFirstVnchw<VT, TRANS_BLOCK>(inBits, transBits, rowLength);

    for (uint32_t g = 0; g < splitNum; ++g) {
        const uint32_t curSplitSize = GetSplitSize(g);
        const uint32_t outputElems = outerReal * curSplitSize;
        const uint32_t outputCopyElems = this->template TransposeSameLenCompactSegment<VT, TRANS_BLOCK>(
            segBits, transBits, outBits, g, curSplitSize, outputElems);

        outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(g) + coreOuterBase * curSplitSize,
                                 coreOuterLength * curSplitSize);
        CopyOutSameLenCompactSegment(inReadyRaw, outerBase * static_cast<uint64_t>(curSplitSize), outputElems,
                                     outputCopyElems);
        SplitVSync<HardEvent::MTE3_V>();
    }
    queue.FreeTensor(inReadyRaw);
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompact<T>::ProcessChunkTaskTyped(uint64_t taskIdx)
{
    const uint64_t outerTileIdx = taskIdx / colChunkNum;
    const uint32_t colChunkIdx = static_cast<uint32_t>(taskIdx % colChunkNum);
    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint64_t localOuterBase = outerBase - coreOuterBase;
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

    const uint32_t splitBase = colChunkIdx * chunkSplitNum;
    if (splitBase >= splitNum) {
        return;
    }
    const uint32_t remainSplit = splitNum - splitBase;
    const uint32_t curChunkSplitNum = remainSplit > chunkSplitNum ? chunkSplitNum : remainSplit;
    const uint32_t colBase = splitBase * splitSize;
    const uint32_t lastSplitInChunk = splitBase + curChunkSplitNum - 1;
    const uint32_t chunkCols = (curChunkSplitNum - 1) * splitSize + GetSplitSize(lastSplitInChunk);

    LocalTensor<T> inLocalRaw = queue.template AllocTensor<T>();
    const uint32_t chunkBytes = chunkCols * sizeof(T);
    const uint32_t alignedChunkBytes = SplitVAlignUp(chunkBytes, BLOCK_SIZE);
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = chunkBytes;
    copyInParam.srcStride = (rowLength - chunkCols) * sizeof(T);
    copyInParam.dstStride = (rowPitch * sizeof(T) - alignedChunkBytes) / BLOCK_SIZE;
    copyInPadParam.isPad = alignedChunkBytes != chunkBytes;
    copyInPadParam.leftPadding = 0;
    copyInPadParam.rightPadding = static_cast<uint8_t>((alignedChunkBytes - chunkBytes) / sizeof(T));
    copyInPadParam.paddingValue = 0;
    DataCopyPad(inLocalRaw, inputGm[localOuterBase * static_cast<uint64_t>(rowLength) + colBase], copyInParam,
                copyInPadParam);
    queue.template EnQue<T>(inLocalRaw);

    LocalTensor<T> inReadyRaw = queue.template DeQue<T>();
    SplitVSync<HardEvent::MTE2_V>();
    LocalTensor<VT> inBits = inReadyRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> transBits = transTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> segBits = segTensor.template ReinterpretCast<VT>();
    LocalTensor<VT> outBits = inReadyRaw.template ReinterpretCast<VT>();

    SplitVFirstVnchw<VT, TRANS_BLOCK>(inBits, transBits, rowPitch);

    for (uint32_t localSplit = 0; localSplit < curChunkSplitNum; ++localSplit) {
        const uint32_t globalSplit = splitBase + localSplit;
        const uint32_t curSplitSize = GetSplitSize(globalSplit);
        const uint32_t outputElems = outerReal * curSplitSize;
        const uint32_t outputCopyElems = SplitVAlignUp(outputElems, alignedNum);
        const uint32_t srcOffset = localSplit * splitSize * TRANS_BLOCK;
        segCopyParam.blockLen = curSplitSize;
        segCopyParam.srcStride = rowPitch - curSplitSize;
        SplitVSync<HardEvent::V_MTE2>();
        DataCopy(segBits, transBits[srcOffset], segCopyParam);
        SplitVSync<HardEvent::MTE2_V>();

        SplitVSecondVnchw<VT, TRANS_BLOCK>(segBits, outBits, curSplitSize);
        SplitVSync<HardEvent::V_MTE3>();

        outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(globalSplit) + coreOuterBase * curSplitSize,
                                 coreOuterLength * curSplitSize);
        const uint32_t outputCopyBytes = outputElems * sizeof(T);
        if (outputCopyBytes <= 65535U && outputCopyElems == outputElems) {
            DataCopy(outputGm[localOuterBase * static_cast<uint64_t>(curSplitSize)], inReadyRaw, outputCopyElems);
        } else {
            copyInParam.blockCount = 1;
            copyInParam.blockLen = outputCopyBytes;
            copyInParam.srcStride = 0;
            copyInParam.dstStride = 0;
            DataCopyPad(outputGm[localOuterBase * static_cast<uint64_t>(curSplitSize)], inReadyRaw, copyInParam);
        }
        SplitVSync<HardEvent::MTE3_V>();
    }
    queue.FreeTensor(inReadyRaw);
}

template <typename T>
__aicore__ inline void SplitVSameLenCompact<T>::Process()
{
    if (outerTile != OUTER_TILE_FIXED || rowLength == 0 || splitSize == 0 || splitNum == 0) {
        return;
    }

    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t tileIdx = loopOff + i;
        const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
        const uint64_t localOuterBase = outerBase - coreOuterBase;
        const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            if (colChunkNum > 1) {
                ProcessChunkTaskTyped<half>(tileIdx);
            } else {
                ProcessTileTyped<half>(localOuterBase, outerReal);
            }
        } else if constexpr (AscendC::Std::is_same<T, uint16_t>::value) {
            if (colChunkNum > 1) {
                ProcessChunkTaskTyped<int16_t>(tileIdx);
            } else {
                ProcessTileTyped<int16_t>(localOuterBase, outerReal);
            }
        } else {
            if (colChunkNum > 1) {
                ProcessChunkTaskTyped<T>(tileIdx);
            } else {
                ProcessTileTyped<T>(localOuterBase, outerReal);
            }
        }
    }
}

#endif // OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_H_
