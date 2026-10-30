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
 * \file split_v_same_len_inner_copy.h
 * \brief Pure copy same-length split for innerLength > 1.
 */

#ifndef OP_KERNEL_SPLIT_V_SAME_LEN_INNER_COPY_H_
#define OP_KERNEL_SPLIT_V_SAME_LEN_INNER_COPY_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVSameLenInnerCopy : private SplitVInnerCopyState<T, 1> {
public:
    __aicore__ inline SplitVSameLenInnerCopy() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataSameLenInnerCopy* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t QUEUE_DEPTH = 1;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

    using Base = SplitVInnerCopyState<T, QUEUE_DEPTH>;
    SPLIT_V_USE_INNER_COPY_STATE(Base);

    uint32_t splitSize;
    uint32_t tailSplitSize;
    uint32_t maxSplitSize;
    uint32_t splitNum;
    uint32_t innerTile;
    uint32_t innerTileNum;
    uint32_t innerTail;
    uint64_t coreOuterBase;
    uint64_t coreOuterLength;

    __aicore__ inline uint64_t AlignUpBytes(uint64_t bytes) const;
    __aicore__ inline uint32_t GetSplitSize(uint32_t splitIdx) const;
    __aicore__ inline void ProcessFullRowPack(uint64_t taskIdx);
    __aicore__ inline void ProcessSegmentInnerPack(uint64_t taskIdx);
    __aicore__ inline void ProcessMidTilePack(uint64_t taskIdx);
    __aicore__ inline void ProcessInnerTilePack(uint64_t taskIdx);
    __aicore__ inline void ProcessSplitChunkPack(uint32_t splitIdx, uint32_t chunkIdx, uint64_t outerTileIdx);
    __aicore__ inline void CopyMidBlock(uint32_t splitIdx, uint64_t localOuterBase, uint32_t outerReal,
                                        uint32_t curSplitSize, uint64_t midOffset, uint32_t midReal, uint64_t dstOffset,
                                        uint32_t dstStride);
};

template <typename T>
__aicore__ inline uint64_t SplitVSameLenInnerCopy<T>::AlignUpBytes(uint64_t bytes) const
{
    return ((bytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
}

template <typename T>
__aicore__ inline uint32_t SplitVSameLenInnerCopy<T>::GetSplitSize(uint32_t splitIdx) const
{
    return (splitIdx + 1 == splitNum) ? tailSplitSize : splitSize;
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::Init(GM_ADDR input, GM_ADDR output,
                                                       SplitVTilingDataSameLenInnerCopy* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    outList.Init((__gm__ void*)output);

    outerLength = tiling->outerLength;
    midLength = tiling->midLength;
    innerLength = tiling->innerLength;
    splitSize = tiling->splitSize;
    tailSplitSize = tiling->tailSplitSize == 0 ? splitSize : tiling->tailSplitSize;
    maxSplitSize = tailSplitSize > splitSize ? tailSplitSize : splitSize;
    splitNum = tiling->splitNum;
    mode = tiling->mode;
    outerTile = tiling->outerTile;
    outerTileNum = tiling->outerTileNum;
    outerTail = tiling->outerTail;
    midTile = tiling->midTile;
    midTileNum = tiling->midTileNum;
    midTail = tiling->midTail;
    innerTile = tiling->innerTile;
    innerTileNum = tiling->innerTileNum;
    innerTail = tiling->innerTail;
    chunkElems = tiling->chunkElems;
    chunkElemsAligned = tiling->chunkElemsAligned;
    chunkNumMax = tiling->chunkNumMax;

    const SplitVTaskRange taskRange = SplitVCalcTaskRange(blockIdx, tiling->formerNum, tiling->formerTaskNum,
                                                          tiling->tailTaskNum);
    taskNum = taskRange.count;
    taskOffset = taskRange.base;

    // Every mode orders tasks by outer tile, so each core can use one contiguous outer window.
    if (taskNum == 0 || outerTileNum == 0) {
        taskNum = 0;
        return;
    }
    const uint64_t tasksPerOuter = tiling->totalTaskNum / outerTileNum;
    if (tasksPerOuter == 0) {
        taskNum = 0;
        return;
    }
    const uint64_t firstOuterTile = taskOffset / tasksPerOuter;
    const uint64_t lastOuterTile = (taskOffset + taskNum - 1) / tasksPerOuter;
    coreOuterBase = firstOuterTile * static_cast<uint64_t>(outerTile);
    const uint64_t coreOuterEnd = (lastOuterTile == static_cast<uint64_t>(outerTileNum - 1)) ?
                                      outerLength :
                                      (lastOuterTile + 1) * static_cast<uint64_t>(outerTile);
    coreOuterLength = coreOuterEnd - coreOuterBase;

    const uint64_t rowElems = midLength * innerLength;
    inputGm.SetGlobalBuffer((__gm__ T*)input + coreOuterBase * rowElems, coreOuterLength * rowElems);

    uint64_t tileBytes = 0;
    if (mode == splitVSameLenInnerCopyFullRowPack) {
        tileBytes = midLength * innerLength * sizeof(T);
    } else if (mode == splitVSameLenInnerCopySegmentInnerPack) {
        tileBytes = static_cast<uint64_t>(maxSplitSize) * innerLength * sizeof(T);
    } else if (mode == splitVSameLenInnerCopySplitChunkPack) {
        tileBytes = static_cast<uint64_t>(chunkElems) * sizeof(T);
    } else if (mode == splitVSameLenInnerCopyMidTilePack) {
        tileBytes = static_cast<uint64_t>(midTile) * innerLength * sizeof(T);
    } else {
        tileBytes = static_cast<uint64_t>(innerTile) * sizeof(T);
    }
    pipe->InitBuffer(queue, DOUBLE_BUFFER_NUM, static_cast<uint32_t>(outerTile * AlignUpBytes(tileBytes)));
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::Process()
{
    for (uint64_t i = 0; i < taskNum; ++i) {
        const uint64_t taskIdx = taskOffset + i;
        if (mode == splitVSameLenInnerCopyFullRowPack) {
            ProcessFullRowPack(taskIdx);
        } else if (mode == splitVSameLenInnerCopySegmentInnerPack) {
            ProcessSegmentInnerPack(taskIdx);
        } else if (mode == splitVSameLenInnerCopySplitChunkPack) {
            const uint32_t chunkIdx = static_cast<uint32_t>(taskIdx % chunkNumMax);
            uint64_t remain = taskIdx / chunkNumMax;
            const uint32_t splitIdx = static_cast<uint32_t>(remain % splitNum);
            const uint64_t outerTileIdx = remain / splitNum;
            ProcessSplitChunkPack(splitIdx, chunkIdx, outerTileIdx);
        } else if (mode == splitVSameLenInnerCopyMidTilePack) {
            ProcessMidTilePack(taskIdx);
        } else {
            ProcessInnerTilePack(taskIdx);
        }
    }
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::CopyMidBlock(uint32_t splitIdx, uint64_t localOuterBase,
                                                               uint32_t outerReal, uint32_t curSplitSize,
                                                               uint64_t midOffset, uint32_t midReal, uint64_t dstOffset,
                                                               uint32_t dstStride)
{
    const uint64_t blockElems = static_cast<uint64_t>(midReal) * innerLength;
    const uint32_t blockBytes = static_cast<uint32_t>(blockElems * sizeof(T));

    LocalTensor<T> inLocal = queue.template AllocTensor<T>();
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = blockBytes;
    copyInParam.srcStride = static_cast<uint32_t>((midLength - midReal) * innerLength * sizeof(T));
    copyInParam.dstStride = 0;
    DataCopyPad(inLocal, inputGm[(localOuterBase * midLength + midOffset) * innerLength], copyInParam, padParam);
    queue.template EnQue<T>(inLocal);

    LocalTensor<T> data = queue.template DeQue<T>();
    outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx) +
                                 coreOuterBase * static_cast<uint64_t>(curSplitSize) * innerLength,
                             coreOuterLength * static_cast<uint64_t>(curSplitSize) * innerLength);
    copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
    copyOutParam.blockLen = blockBytes;
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = dstStride;
    DataCopyPad(outputGm[dstOffset], data, copyOutParam);
    queue.FreeTensor(data);
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::ProcessFullRowPack(uint64_t taskIdx)
{
    const uint64_t outerBase = taskIdx * static_cast<uint64_t>(outerTile);
    const uint64_t localOuterBase = outerBase - coreOuterBase;
    const uint32_t outerReal = (taskIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint64_t rowElems = midLength * innerLength;
    const uint32_t rowBytes = static_cast<uint32_t>(rowElems * sizeof(T));

    LocalTensor<T> inLocal = queue.template AllocTensor<T>();
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = rowBytes;
    copyInParam.srcStride = 0;
    copyInParam.dstStride = 0;
    DataCopyPad(inLocal, inputGm[localOuterBase * rowElems], copyInParam, padParam);
    queue.template EnQue<T>(inLocal);

    LocalTensor<T> data = queue.template DeQue<T>();
    DataCopyParams fullRowOutParam{0, 0, 0, 0};
    fullRowOutParam.blockCount = static_cast<uint16_t>(outerReal);
    fullRowOutParam.dstStride = 0;
    for (uint32_t splitIdx = 0; splitIdx < splitNum; ++splitIdx) {
        const uint32_t curSplitSize = GetSplitSize(splitIdx);
        const uint32_t splitBytes = static_cast<uint32_t>(static_cast<uint64_t>(curSplitSize) * innerLength *
                                                          sizeof(T));
        const uint16_t gapBlockLen = static_cast<uint16_t>((rowBytes - splitBytes) / BLOCK_SIZE);
        fullRowOutParam.blockLen = static_cast<uint16_t>(splitBytes);
        fullRowOutParam.srcStride = gapBlockLen;
        outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx) +
                                     coreOuterBase * static_cast<uint64_t>(curSplitSize) * innerLength,
                                 coreOuterLength * static_cast<uint64_t>(curSplitSize) * innerLength);
        const uint64_t localOffset = static_cast<uint64_t>(splitIdx) * splitSize * innerLength;
        DataCopyPad(outputGm[localOuterBase * curSplitSize * innerLength], data[localOffset], fullRowOutParam);
    }
    queue.FreeTensor(data);
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::ProcessSegmentInnerPack(uint64_t taskIdx)
{
    const uint64_t outerBase = taskIdx * static_cast<uint64_t>(outerTile);
    const uint64_t localOuterBase = outerBase - coreOuterBase;
    const uint32_t outerReal = (taskIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

    for (uint32_t splitIdx = 0; splitIdx < splitNum; ++splitIdx) {
        const uint32_t curSplitSize = GetSplitSize(splitIdx);
        const uint64_t blockElems = static_cast<uint64_t>(curSplitSize) * innerLength;
        const uint64_t midOffset = static_cast<uint64_t>(splitIdx) * splitSize;
        CopyMidBlock(splitIdx, localOuterBase, outerReal, curSplitSize, midOffset, curSplitSize,
                     localOuterBase * blockElems, 0);
    }
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::ProcessMidTilePack(uint64_t taskIdx)
{
    uint64_t remain = taskIdx;
    const uint32_t midTileIdx = static_cast<uint32_t>(remain % midTileNum);
    remain /= midTileNum;
    const uint32_t splitIdx = static_cast<uint32_t>(remain % splitNum);
    const uint64_t outerTileIdx = remain / splitNum;

    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint64_t localOuterBase = outerBase - coreOuterBase;
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint32_t curSplitSize = GetSplitSize(splitIdx);
    const uint64_t splitMidOffset = static_cast<uint64_t>(midTileIdx) * midTile;
    if (splitMidOffset >= curSplitSize) {
        return;
    }
    const uint32_t remainMid = static_cast<uint32_t>(curSplitSize - splitMidOffset);
    const uint32_t midReal = remainMid > midTile ? midTile : remainMid;
    const uint64_t midOffset = static_cast<uint64_t>(splitIdx) * splitSize + splitMidOffset;
    const uint64_t dstOffset = (localOuterBase * curSplitSize + splitMidOffset) * innerLength;
    const uint32_t dstStride = static_cast<uint32_t>((curSplitSize - midReal) * innerLength * sizeof(T));
    CopyMidBlock(splitIdx, localOuterBase, outerReal, curSplitSize, midOffset, midReal, dstOffset, dstStride);
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::ProcessSplitChunkPack(uint32_t splitIdx, uint32_t chunkIdx,
                                                                        uint64_t outerTileIdx)
{
    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint64_t localOuterBase = outerBase - coreOuterBase;
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint32_t curSplitSize = GetSplitSize(splitIdx);
    const uint64_t splitElems = static_cast<uint64_t>(curSplitSize) * innerLength;
    const uint64_t chunkOffset = static_cast<uint64_t>(chunkIdx) * chunkElems;
    const SplitVChunkCopyInfo chunkInfo = SplitVMakeChunkCopyInfo(splitElems, chunkOffset, chunkElems, sizeof(T),
                                                                  BLOCK_SIZE);
    if (!chunkInfo.valid) {
        return;
    }
    const uint64_t splitOffset = static_cast<uint64_t>(splitIdx) * splitSize;

    LocalTensor<T> data = CopyInInnerChunk((localOuterBase * midLength + splitOffset) * innerLength + chunkOffset,
                                           outerReal, chunkInfo);
    outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx) + coreOuterBase * splitElems,
                             coreOuterLength * splitElems);
    copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
    copyOutParam.blockLen = chunkInfo.copyBytes;
    copyOutParam.srcStride = chunkInfo.localPitchBytes - chunkInfo.copyBytes;
    copyOutParam.dstStride = static_cast<uint32_t>((splitElems - chunkInfo.copyElems) * sizeof(T));
    DataCopyPad(outputGm[localOuterBase * splitElems + chunkOffset], data, copyOutParam);
    queue.FreeTensor(data);
}

template <typename T>
__aicore__ inline void SplitVSameLenInnerCopy<T>::ProcessInnerTilePack(uint64_t taskIdx)
{
    uint64_t remain = taskIdx;
    const uint32_t innerTileIdx = static_cast<uint32_t>(remain % innerTileNum);
    remain /= innerTileNum;
    const uint32_t splitMidIdx = static_cast<uint32_t>(remain % midTileNum);
    remain /= midTileNum;
    const uint32_t splitIdx = static_cast<uint32_t>(remain % splitNum);
    const uint64_t outerTileIdx = remain / splitNum;

    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint64_t localOuterBase = outerBase - coreOuterBase;
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint32_t innerReal = (innerTileIdx == innerTileNum - 1) ? innerTail : innerTile;
    const uint32_t curSplitSize = GetSplitSize(splitIdx);
    if (splitMidIdx >= curSplitSize) {
        return;
    }
    const uint64_t innerOffset = static_cast<uint64_t>(innerTileIdx) * innerTile;
    const uint32_t blockBytes = static_cast<uint32_t>(static_cast<uint64_t>(innerReal) * sizeof(T));
    const uint64_t inputMid = static_cast<uint64_t>(splitIdx) * splitSize + splitMidIdx;

    LocalTensor<T> inLocal = queue.template AllocTensor<T>();
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = blockBytes;
    copyInParam.srcStride = static_cast<uint32_t>((midLength * innerLength - innerReal) * sizeof(T));
    copyInParam.dstStride = 0;
    DataCopyPad(inLocal, inputGm[(localOuterBase * midLength + inputMid) * innerLength + innerOffset], copyInParam,
                padParam);
    queue.template EnQue<T>(inLocal);

    LocalTensor<T> data = queue.template DeQue<T>();
    outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx) +
                                 coreOuterBase * static_cast<uint64_t>(curSplitSize) * innerLength,
                             coreOuterLength * static_cast<uint64_t>(curSplitSize) * innerLength);
    copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
    copyOutParam.blockLen = blockBytes;
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = static_cast<uint32_t>((static_cast<uint64_t>(curSplitSize) * innerLength - innerReal) *
                                                   sizeof(T));
    const uint64_t dstOffset = (localOuterBase * curSplitSize + splitMidIdx) * innerLength + innerOffset;
    DataCopyPad(outputGm[dstOffset], data, copyOutParam);
    queue.FreeTensor(data);
}

#endif // OP_KERNEL_SPLIT_V_SAME_LEN_INNER_COPY_H_
