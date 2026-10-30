/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_DOUBLE_BUFFER_H_
#define OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_DOUBLE_BUFFER_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVSameLenCompactDoubleBuffer {
public:
    __aicore__ inline SplitVSameLenCompactDoubleBuffer() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataSameLenCompact* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t TRANS_BLOCK = 16;
    static constexpr uint32_t OUTER_TILE_FIXED = 256;
    static constexpr uint32_t QUEUE_DEPTH = 1;
    static constexpr uint32_t SINGLE_BUFFER_NUM = 1;
    static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    TQue<TPosition::VECIN, QUEUE_DEPTH> inputQueue;
    TQue<TPosition::VECIN, QUEUE_DEPTH> segQueue;
    TQue<TPosition::VECOUT, QUEUE_DEPTH> outputQueue;
    TBuf<TPosition::VECCALC> transBuf;
    LocalTensor<T> transTensor;

    SPLIT_V_COMPACT_CORE_FIELDS;

    DataCopyExtParams copyParam{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> copyInPadParam{false, 0, 0, 0};
    DataCopyParams segCopyParam{0, 0, 0, 0};

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const;
    __aicore__ inline uint32_t GetSplitSize(uint32_t splitIdx) const;
    template <typename VT>
    __aicore__ inline void FirstVnchw(LocalTensor<VT> src, LocalTensor<VT> dst);
    template <typename VT>
    __aicore__ inline void SecondVnchw(LocalTensor<VT> src, LocalTensor<VT> dst, uint32_t curSplitSize);
    __aicore__ inline void WaitVToMte2();
    __aicore__ inline void CopySeg(uint32_t splitIdx);
    template <typename VT>
    __aicore__ inline void ComputeSplit(uint32_t splitIdx);
    __aicore__ inline void CopyOut(uint32_t splitIdx, uint64_t outerBase, uint32_t outerReal);
    template <typename VT>
    __aicore__ inline void ProcessTileTyped(uint64_t outerBase, uint32_t outerReal);
};

template <typename T>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::Init(GM_ADDR input, GM_ADDR output,
                                                                 SplitVTilingDataSameLenCompact* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    if (blockIdx >= GetBlockNum()) {
        return;
    }

    outList.Init((__gm__ void*)output);
    outerLength = tiling->outerLength;
    rowLength = tiling->rowLength;
    splitSize = tiling->splitSize;
    tailSplitSize = tiling->tailSplitSize == 0 ? splitSize : tiling->tailSplitSize;
    splitNum = tiling->splitNum;
    outerTile = tiling->outerTile;
    outerTail = tiling->outerTail;
    outerTileNum = tiling->outerTileNum;
    alignedNum = BLOCK_SIZE / sizeof(T);

    if (blockIdx < tiling->formerNum) {
        tileNum = tiling->formerOuterTileNum;
        loopOff = static_cast<uint64_t>(blockIdx) * tiling->formerOuterTileNum;
    } else {
        tileNum = tiling->tailOuterTileNum;
        loopOff = static_cast<uint64_t>(tiling->formerNum) * tiling->formerOuterTileNum +
                  static_cast<uint64_t>(blockIdx - tiling->formerNum) * tiling->tailOuterTileNum;
    }

    coreOuterBase = loopOff * outerTile;
    const uint64_t lastOuterTile = loopOff + tileNum - 1;
    const uint64_t coreOuterEnd = lastOuterTile == static_cast<uint64_t>(outerTileNum - 1) ?
                                      outerLength :
                                      (lastOuterTile + 1) * outerTile;
    coreOuterLength = coreOuterEnd - coreOuterBase;
    inputGm.SetGlobalBuffer((__gm__ T*)input + coreOuterBase * rowLength, coreOuterLength * rowLength);

    const uint32_t inputBytes = outerTile * rowLength * sizeof(T);
    const uint32_t splitBytes = outerTile * splitSize * sizeof(T);
    pipe->InitBuffer(inputQueue, SINGLE_BUFFER_NUM, inputBytes);
    pipe->InitBuffer(segQueue, DOUBLE_BUFFER_NUM, splitBytes);
    pipe->InitBuffer(outputQueue, DOUBLE_BUFFER_NUM, splitBytes);
    pipe->InitBuffer(transBuf, inputBytes);
    transTensor = transBuf.Get<T>();

    segCopyParam.blockCount = TRANS_BLOCK;
    segCopyParam.blockLen = splitSize;
    segCopyParam.srcStride = rowLength - splitSize;
    segCopyParam.dstStride = 0;
}

template <typename T>
__aicore__ inline uint32_t SplitVSameLenCompactDoubleBuffer<T>::AlignUp(uint32_t value, uint32_t align) const
{
    return SplitVAlignUp(value, align);
}

template <typename T>
__aicore__ inline uint32_t SplitVSameLenCompactDoubleBuffer<T>::GetSplitSize(uint32_t splitIdx) const
{
    return splitIdx + 1 == splitNum ? tailSplitSize : splitSize;
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::FirstVnchw(LocalTensor<VT> src, LocalTensor<VT> dst)
{
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = static_cast<uint8_t>(rowLength);
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : TRANS_BLOCK;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;

    uint64_t srcList[TRANS_BLOCK];
    uint64_t dstList[TRANS_BLOCK];
    for (uint32_t j = 0; j < TRANS_BLOCK; ++j) {
        srcList[j] = reinterpret_cast<uint64_t>(src[static_cast<uint32_t>(j) * TRANS_BLOCK * rowLength].GetPhyAddr());
        dstList[j] = reinterpret_cast<uint64_t>(dst[static_cast<uint32_t>(j) * TRANS_BLOCK].GetPhyAddr());
    }
    TransDataTo5HD<VT>(dstList, srcList, transDataParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::SecondVnchw(LocalTensor<VT> src, LocalTensor<VT> dst,
                                                                        uint32_t curSplitSize)
{
    SplitVSecondVnchw<VT, TRANS_BLOCK>(src, dst, curSplitSize);
}

template <typename T>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::WaitVToMte2()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
}

template <typename T>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::CopySeg(uint32_t splitIdx)
{
    const uint32_t curSplitSize = GetSplitSize(splitIdx);
    const uint32_t srcOffset = splitIdx * splitSize * TRANS_BLOCK;
    LocalTensor<T> segLocal = segQueue.AllocTensor<T>();
    segCopyParam.blockLen = curSplitSize;
    segCopyParam.srcStride = rowLength - curSplitSize;
    DataCopy(segLocal, transTensor[srcOffset], segCopyParam);
    segQueue.EnQue<T>(segLocal);
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::ComputeSplit(uint32_t splitIdx)
{
    const uint32_t curSplitSize = GetSplitSize(splitIdx);
    LocalTensor<T> segLocalRaw = segQueue.DeQue<T>();
    LocalTensor<T> outputLocalRaw = outputQueue.AllocTensor<T>();
    LocalTensor<VT> segBits = segLocalRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> outputBits = outputLocalRaw.template ReinterpretCast<VT>();
    SecondVnchw<VT>(segBits, outputBits, curSplitSize);
    segQueue.FreeTensor(segLocalRaw);
    outputQueue.EnQue<T>(outputLocalRaw);
}

template <typename T>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::CopyOut(uint32_t splitIdx, uint64_t outerBase,
                                                                    uint32_t outerReal)
{
    const uint32_t curSplitSize = GetSplitSize(splitIdx);
    const uint32_t outputElems = outerReal * curSplitSize;
    const uint32_t outputCopyElems = AlignUp(outputElems, alignedNum);
    LocalTensor<T> outputLocal = outputQueue.DeQue<T>();

    outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(splitIdx) + coreOuterBase * curSplitSize,
                             coreOuterLength * curSplitSize);
    const uint32_t outputCopyBytes = outputElems * sizeof(T);
    if (outputCopyBytes <= 65535U && outputCopyElems == outputElems) {
        DataCopy(outputGm[outerBase * static_cast<uint64_t>(curSplitSize)], outputLocal, outputCopyElems);
    } else {
        copyParam.blockCount = 1;
        copyParam.blockLen = outputCopyBytes;
        copyParam.srcStride = 0;
        copyParam.dstStride = 0;
        DataCopyPad(outputGm[outerBase * static_cast<uint64_t>(curSplitSize)], outputLocal, copyParam);
    }
    outputQueue.FreeTensor(outputLocal);
}

template <typename T>
template <typename VT>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::ProcessTileTyped(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
    const uint32_t inputElems = outerReal * rowLength;
    const uint32_t inputCopyBytes = inputElems * sizeof(T);
    if (outerReal == outerTile && inputCopyBytes <= 65535U) {
        DataCopy(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], inputElems);
    } else {
        copyParam.blockCount = 1;
        copyParam.blockLen = inputCopyBytes;
        copyParam.srcStride = 0;
        copyParam.dstStride = 0;
        DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyParam, copyInPadParam);
    }
    inputQueue.EnQue<T>(inputLocal);

    LocalTensor<T> inputReadyRaw = inputQueue.DeQue<T>();
    LocalTensor<VT> inputBits = inputReadyRaw.template ReinterpretCast<VT>();
    LocalTensor<VT> transBits = transTensor.template ReinterpretCast<VT>();
    FirstVnchw<VT>(inputBits, transBits);
    WaitVToMte2();
    inputQueue.FreeTensor(inputReadyRaw);

    CopySeg(0);
    for (uint32_t splitIdx = 0; splitIdx < splitNum; ++splitIdx) {
        ComputeSplit<VT>(splitIdx);
        if (splitIdx + 1 < splitNum) {
            CopySeg(splitIdx + 1);
        }
        CopyOut(splitIdx, outerBase, outerReal);
    }
}

template <typename T>
__aicore__ inline void SplitVSameLenCompactDoubleBuffer<T>::Process()
{
    if (outerTile != OUTER_TILE_FIXED || rowLength == 0 || rowLength > 128 || splitSize == 0 || splitNum < 2) {
        return;
    }

    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t tileIdx = loopOff + i;
        const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
        const uint64_t localOuterBase = outerBase - coreOuterBase;
        const uint32_t outerReal = tileIdx == static_cast<uint64_t>(outerTileNum - 1) ? outerTail : outerTile;
        ProcessTileTyped<half>(localOuterBase, outerReal);
    }
}

#endif // OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_DOUBLE_BUFFER_H_
