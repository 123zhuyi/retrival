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
 * \file split_v_uneven_len_inner_copy.h
 * \brief Pure copy uneven-length split for innerLength > 1.
 */

#ifndef OP_KERNEL_SPLIT_V_UNEVEN_LEN_INNER_COPY_H_
#define OP_KERNEL_SPLIT_V_UNEVEN_LEN_INNER_COPY_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVUnevenLenInnerCopy : private SplitVInnerCopyState<T, 1> {
public:
    __aicore__ inline SplitVUnevenLenInnerCopy() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output,
                                __tiling_data_ptr__ SplitVTilingDataUnevenInnerAlignedMid* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t QUEUE_DEPTH = 1;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

    using Base = SplitVInnerCopyState<T, QUEUE_DEPTH>;
    SPLIT_V_USE_INNER_COPY_STATE(Base);

    uint32_t splitNum;

    uint32_t sizeSplits[maxSplitNum];
    uint32_t splitOffsets[maxSplitNum];

    __aicore__ inline uint64_t AlignUpBytes(uint64_t bytes) const;
    __aicore__ inline void WaitMte2ToMte3();
    __aicore__ inline void WaitMte3ToMte2();
    __aicore__ inline void ProcessAlignedMidTile(uint32_t midTileIdx, uint64_t outerTileIdx);
    __aicore__ inline void ProcessSegmentPack(uint32_t splitIdx, uint64_t outerTileIdx);
    __aicore__ inline void ProcessSplitChunkPack(uint32_t splitIdx, uint32_t chunkIdx, uint64_t outerTileIdx);
};

template <typename T>
__aicore__ inline uint64_t SplitVUnevenLenInnerCopy<T>::AlignUpBytes(uint64_t bytes) const
{
    return (bytes + BLOCK_SIZE - 1) & ~(static_cast<uint64_t>(BLOCK_SIZE) - 1);
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::WaitMte2ToMte3()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::WaitMte3ToMte2()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::Init(
    GM_ADDR input, GM_ADDR output, __tiling_data_ptr__ SplitVTilingDataUnevenInnerAlignedMid* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    outList.Init((__gm__ void*)output);

    outerLength = tiling->outerLength;
    midLength = tiling->midLength;
    innerLength = tiling->innerLength;
    splitNum = tiling->splitNum;
    mode = tiling->mode;
    outerTile = tiling->outerTile;
    outerTileNum = tiling->outerTileNum;
    outerTail = tiling->outerTail;
    midTile = tiling->midTile;
    midTileNum = tiling->midTileNum;
    midTail = tiling->midTail;
    chunkElems = tiling->chunkElems;
    chunkElemsAligned = tiling->chunkElemsAligned;
    chunkNumMax = tiling->chunkNumMax;

    for (uint32_t i = 0; i < splitNum; ++i) {
        sizeSplits[i] = tiling->sizeSplits[i];
        splitOffsets[i] = tiling->splitOffsets[i];
    }

    const SplitVTaskRange taskRange = SplitVCalcTaskRange(blockIdx, tiling->formerNum, tiling->formerTaskNum,
                                                          tiling->tailTaskNum);
    taskNum = taskRange.count;
    taskOffset = taskRange.base;

    inputGm.SetGlobalBuffer((__gm__ T*)input, tiling->totalLength);
    const uint64_t tileBytes = mode == splitVUnevenInnerSplitChunkPack ?
                                   static_cast<uint64_t>(chunkElems) * sizeof(T) :
                                   static_cast<uint64_t>(midTile) * innerLength * sizeof(T);
    pipe->InitBuffer(queue, DOUBLE_BUFFER_NUM, static_cast<uint32_t>(outerTile * AlignUpBytes(tileBytes)));
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::Process()
{
    if (mode == splitVUnevenInnerSegmentPack) {
        uint32_t splitIdx = static_cast<uint32_t>(taskOffset % splitNum);
        uint64_t outerTileIdx = taskOffset / splitNum;
        for (uint64_t i = 0; i < taskNum; ++i) {
            ProcessSegmentPack(splitIdx, outerTileIdx);
            ++splitIdx;
            if (splitIdx == splitNum) {
                splitIdx = 0;
                ++outerTileIdx;
            }
        }
    } else if (mode == splitVUnevenInnerSplitChunkPack) {
        uint32_t chunkIdx = static_cast<uint32_t>(taskOffset % chunkNumMax);
        uint64_t remain = taskOffset / chunkNumMax;
        uint32_t splitIdx = static_cast<uint32_t>(remain % splitNum);
        uint64_t outerTileIdx = remain / splitNum;
        for (uint64_t i = 0; i < taskNum; ++i) {
            ProcessSplitChunkPack(splitIdx, chunkIdx, outerTileIdx);
            ++chunkIdx;
            if (chunkIdx == chunkNumMax) {
                chunkIdx = 0;
                ++splitIdx;
                if (splitIdx == splitNum) {
                    splitIdx = 0;
                    ++outerTileIdx;
                }
            }
        }
    } else {
        uint32_t midTileIdx = static_cast<uint32_t>(taskOffset % midTileNum);
        uint64_t outerTileIdx = taskOffset / midTileNum;
        for (uint64_t i = 0; i < taskNum; ++i) {
            ProcessAlignedMidTile(midTileIdx, outerTileIdx);
            ++midTileIdx;
            if (midTileIdx == midTileNum) {
                midTileIdx = 0;
                ++outerTileIdx;
            }
        }
    }
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::ProcessSegmentPack(uint32_t splitIdx, uint64_t outerTileIdx)
{
    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint64_t splitSize = sizeSplits[splitIdx];
    const uint64_t splitBegin = splitOffsets[splitIdx];
    const uint32_t blockBytes = static_cast<uint32_t>(splitSize * innerLength * sizeof(T));

    LocalTensor<T> inLocal = queue.template AllocTensor<T>();
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = blockBytes;
    copyInParam.srcStride = static_cast<uint32_t>((midLength - splitSize) * innerLength * sizeof(T));
    copyInParam.dstStride = 0;
    DataCopyPad(inLocal, inputGm[(outerBase * midLength + splitBegin) * innerLength], copyInParam, padParam);
    queue.template EnQue<T>(inLocal);

    LocalTensor<T> data = queue.template DeQue<T>();
    outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx), outerLength * splitSize * innerLength);
    copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
    copyOutParam.blockLen = blockBytes;
    copyOutParam.srcStride = 0;
    copyOutParam.dstStride = 0;
    DataCopyPad(outputGm[outerBase * splitSize * innerLength], data, copyOutParam);
    queue.FreeTensor(data);
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::ProcessSplitChunkPack(uint32_t splitIdx, uint32_t chunkIdx,
                                                                          uint64_t outerTileIdx)
{
    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint64_t splitSize = sizeSplits[splitIdx];
    const uint64_t splitBegin = splitOffsets[splitIdx];
    const uint64_t splitElems = splitSize * innerLength;
    const uint64_t chunkOffset = static_cast<uint64_t>(chunkIdx) * chunkElems;
    const SplitVChunkCopyInfo chunkInfo = SplitVMakeChunkCopyInfo(splitElems, chunkOffset, chunkElems, sizeof(T),
                                                                  BLOCK_SIZE);
    if (!chunkInfo.valid) {
        return;
    }

    LocalTensor<T> data = CopyInInnerChunk((outerBase * midLength + splitBegin) * innerLength + chunkOffset, outerReal,
                                           chunkInfo);
    outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx), outerLength * splitSize * innerLength);
    copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
    copyOutParam.blockLen = chunkInfo.copyBytes;
    copyOutParam.srcStride = chunkInfo.localPitchBytes - chunkInfo.copyBytes;
    copyOutParam.dstStride = static_cast<uint32_t>((splitElems - chunkInfo.copyElems) * sizeof(T));
    DataCopyPad(outputGm[outerBase * splitElems + chunkOffset], data, copyOutParam);
    queue.FreeTensor(data);
}

template <typename T>
__aicore__ inline void SplitVUnevenLenInnerCopy<T>::ProcessAlignedMidTile(uint32_t midTileIdx, uint64_t outerTileIdx)
{
    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    const uint32_t midReal = (midTileIdx == midTileNum - 1) ? midTail : midTile;
    const uint64_t midBase = static_cast<uint64_t>(midTileIdx) * midTile;
    const uint64_t blockElems = static_cast<uint64_t>(midReal) * innerLength;
    const uint32_t blockBytes = static_cast<uint32_t>(blockElems * sizeof(T));
    const uint32_t localPitchBytes = static_cast<uint32_t>(AlignUpBytes(blockBytes));

    LocalTensor<T> inLocal = queue.template AllocTensor<T>();
    copyInParam.blockCount = static_cast<uint16_t>(outerReal);
    copyInParam.blockLen = blockBytes;
    copyInParam.srcStride = static_cast<uint32_t>((midLength - midReal) * innerLength * sizeof(T));
    copyInParam.dstStride = 0;
    DataCopyPad(inLocal, inputGm[(outerBase * midLength + midBase) * innerLength], copyInParam, padParam);
    queue.template EnQue<T>(inLocal);

    LocalTensor<T> data = queue.template DeQue<T>();
    WaitMte2ToMte3();
    const uint64_t tileBegin = midBase;
    const uint64_t tileEnd = midBase + midReal;
    for (uint32_t splitIdx = 0; splitIdx < splitNum; ++splitIdx) {
        const uint64_t splitBegin = splitOffsets[splitIdx];
        const uint64_t splitSize = sizeSplits[splitIdx];
        const uint64_t splitEnd = splitBegin + splitSize;
        if (tileEnd <= splitBegin || tileBegin >= splitEnd) {
            continue;
        }
        const uint64_t copyBegin = tileBegin > splitBegin ? tileBegin : splitBegin;
        const uint64_t copyEnd = tileEnd < splitEnd ? tileEnd : splitEnd;
        const uint64_t copyMid = copyEnd - copyBegin;
        const uint32_t copyBytes = static_cast<uint32_t>(copyMid * innerLength * sizeof(T));
        outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIdx),
                                 outerLength * splitSize * innerLength);

        copyOutParam.blockCount = static_cast<uint16_t>(outerReal);
        copyOutParam.blockLen = copyBytes;
        copyOutParam.srcStride = static_cast<uint32_t>(
            (localPitchBytes - static_cast<uint32_t>(AlignUpBytes(copyBytes))) / BLOCK_SIZE);
        copyOutParam.dstStride = static_cast<uint32_t>((splitSize - copyMid) * innerLength * sizeof(T));
        const uint64_t srcOffset = (copyBegin - tileBegin) * innerLength;
        const uint64_t dstOffset = (outerBase * splitSize + (copyBegin - splitBegin)) * innerLength;
        DataCopyPad(outputGm[dstOffset], data[srcOffset], copyOutParam);
    }
    WaitMte3ToMte2();
    queue.FreeTensor(data);
}

#endif // OP_KERNEL_SPLIT_V_UNEVEN_LEN_INNER_COPY_H_
