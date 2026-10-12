/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License. THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the
 * software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_8BIT_H_
#define OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_8BIT_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVSameLenCompact8Bit {
public:
    __aicore__ inline SplitVSameLenCompact8Bit() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR yList, const SplitVTilingDataSameLenCompact* tilingData,
                                TPipe* pipeIn)
    {
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= GetBlockNum()) {
            return;
        }

        pipe = pipeIn;
        totalLength = tilingData->totalLength;
        outerLength = tilingData->outerLength;
        rowLength = tilingData->rowLength;
        splitSize = tilingData->splitSize;
        tailSplitSize = tilingData->tailSplitSize == 0 ? splitSize : tilingData->tailSplitSize;
        splitNum = tilingData->splitNum;
        outerTile = tilingData->outerTile;
        outerTileNum = tilingData->outerTileNum;
        outerTail = tilingData->outerTail;
        rowTransLen = tilingData->rowTransLen;
        splitTransLen = tilingData->splitTransLen;
        chunkSplitNum = tilingData->chunkSplitNum;
        colChunkNum = tilingData->colChunkNum;
        if (blockIdx < tilingData->formerNum) {
            tileNum = tilingData->formerOuterTileNum;
            loopOff = static_cast<uint64_t>(blockIdx) * tilingData->formerOuterTileNum;
        } else {
            tileNum = tilingData->tailOuterTileNum;
            loopOff = static_cast<uint64_t>(tilingData->formerNum) * tilingData->formerOuterTileNum +
                      static_cast<uint64_t>(blockIdx - tilingData->formerNum) * tilingData->tailOuterTileNum;
        }

        inputGm.SetGlobalBuffer((__gm__ T*)x, totalLength);
        outList.Init((__gm__ void*)yList);

        pipe->InitBuffer(inQueue, BUFFER_NUM, GetInputStageBytes());
        pipe->InitBuffer(outQueue, BUFFER_NUM, GetSegmentStageBytes());
        pipe->InitBuffer(transBuffer, GetTransStageBytes());
        transTensor = transBuffer.Get<T>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (!Is8BitType()) {
            return;
        } else {
            if (!IsValidTiling()) {
                return;
            }

            for (uint64_t i = 0; i < tileNum; ++i) {
                if (colChunkNum == 1) {
                    const uint64_t tileIdx = loopOff + i;
                    const uint64_t outerBase = tileIdx * static_cast<uint64_t>(outerTile);
                    const uint32_t outerReal = (tileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail :
                                                                                                      outerTile;
                    if (outerTile == B8_FULL_ROW_256_OUTER_TILE) {
                        ProcessFullRow256(outerBase, outerReal);
                    } else {
                        ProcessTile(outerBase, outerReal);
                    }
                } else {
                    if (IsOddLargePureCopyLayout()) {
                        ProcessPureCopyTask(loopOff + i);
                    } else {
                        ProcessChunkTask(loopOff + i);
                    }
                }
            }
        }
    }

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t B8_ADDR_COUNT = 16;
    static constexpr uint32_t B8_ELEMS_PER_DB = 32;
    static constexpr uint32_t B8_COMPACT_OUTER_TILE = 512;
    static constexpr uint32_t B8_FULL_ROW_256_OUTER_TILE = 256;
    static constexpr uint32_t B8_FULL_ROW_256_GROUP_ROWS = 8;
    static constexpr uint32_t B8_FULL_ROW_512_GROUP_ROWS = 16;

    __aicore__ inline uint32_t GetInputStageBytes() const
    {
        if (IsOddLargePureCopyLayout()) {
            return B8_ELEMS_PER_DB * splitTransLen * sizeof(T);
        }
        return B8_ELEMS_PER_DB * rowTransLen * sizeof(T);
    }

    __aicore__ inline uint32_t GetTransStageBytes() const { return B8_ELEMS_PER_DB * rowTransLen * sizeof(T); }

    __aicore__ inline uint32_t GetSegmentStageBytes() const { return B8_ELEMS_PER_DB * splitTransLen * sizeof(T); }

    __aicore__ inline bool IsValidTiling() const
    {
        if (rowLength == 0 || splitSize == 0 || tailSplitSize == 0 || tailSplitSize > splitSize || splitNum == 0 ||
            chunkSplitNum == 0 || colChunkNum == 0) {
            return false;
        }
        if (outerTile == B8_FULL_ROW_256_OUTER_TILE) {
            return colChunkNum == 1 && chunkSplitNum == splitNum && rowLength > 128 && rowLength % 4 == 0 &&
                   splitSize % 2 == 0 && rowTransLen == B8_FULL_ROW_256_GROUP_ROWS * rowLength &&
                   splitTransLen == B8_FULL_ROW_256_GROUP_ROWS * splitSize &&
                   IsAligned32Bytes(rowTransLen * sizeof(T)) && IsAligned16Bytes(splitTransLen * sizeof(T)) &&
                   rowLength / 4 <= UINT8_MAX && (splitSize + 3) / 4 <= UINT8_MAX;
        }
        if (outerTile != B8_COMPACT_OUTER_TILE) {
            return false;
        }
        if (IsFullRow512ParityLayout()) {
            return (rowLength + 1) / 2 <= UINT8_MAX && (splitSize + 1) / 2 <= UINT8_MAX;
        }
        if (IsOddLargePureCopyLayout()) {
            return splitTransLen >= B8_ADDR_COUNT * splitSize &&
                   IsAligned32Bytes(splitTransLen / B8_ADDR_COUNT * sizeof(T));
        }
        if (IsChunk512ParityLayout()) {
            return (rowTransLen / B8_FULL_ROW_512_GROUP_ROWS) % B8_ELEMS_PER_DB == 0 &&
                   static_cast<uint64_t>(chunkSplitNum) * colChunkNum == splitNum &&
                   (rowTransLen / B8_FULL_ROW_512_GROUP_ROWS + 1) / 2 <= UINT8_MAX && (splitSize + 1) / 2 <= UINT8_MAX;
        }
        return (colChunkNum > 1 || rowTransLen >= B8_ADDR_COUNT * rowLength) &&
               splitTransLen >= B8_ADDR_COUNT * splitSize && IsAligned32Bytes(rowTransLen * sizeof(T)) &&
               IsAligned32Bytes(splitTransLen * sizeof(T)) && rowTransLen / B8_ELEMS_PER_DB <= UINT8_MAX &&
               splitTransLen / B8_ELEMS_PER_DB <= UINT8_MAX;
    }

    template <typename U = T>
    __aicore__ static constexpr bool Is8BitType()
    {
        return IsSameType<U, int8_t>::value || IsSameType<U, uint8_t>::value;
    }

    __aicore__ inline bool IsAligned32Bytes(uint32_t bytes) const
    {
        return SplitVIsAlignedBytes(bytes, B8_ELEMS_PER_DB);
    }

    __aicore__ inline bool IsAligned16Bytes(uint32_t bytes) const { return SplitVIsAlignedBytes(bytes, B8_ADDR_COUNT); }

    __aicore__ inline uint32_t AlignUp32Bytes(uint32_t bytes) const { return SplitVAlignUp(bytes, B8_ELEMS_PER_DB); }

    __aicore__ inline uint32_t GetSplitSize(uint32_t splitIndex) const
    {
        return (splitIndex + 1 == splitNum) ? tailSplitSize : splitSize;
    }

    __aicore__ inline uint32_t GetSecondVnchw512Pitch(uint32_t curSplitSize) const
    {
        return B8_ADDR_COUNT * curSplitSize;
    }

    __aicore__ inline bool IsFullRow512ParityLayout() const
    {
        return outerTile == B8_COMPACT_OUTER_TILE && colChunkNum == 1 && chunkSplitNum == splitNum &&
               rowTransLen == B8_FULL_ROW_512_GROUP_ROWS * rowLength &&
               splitTransLen == B8_FULL_ROW_512_GROUP_ROWS * splitSize && IsAligned16Bytes(rowTransLen * sizeof(T)) &&
               IsAligned16Bytes(splitTransLen * sizeof(T));
    }

    __aicore__ inline bool IsChunk512ParityLayout() const
    {
        const uint32_t chunkColsMax = chunkSplitNum * splitSize;
        return outerTile == B8_COMPACT_OUTER_TILE && colChunkNum > 1 && chunkColsMax != 0 &&
               rowTransLen == B8_FULL_ROW_512_GROUP_ROWS * chunkColsMax &&
               splitTransLen == B8_FULL_ROW_512_GROUP_ROWS * splitSize && IsAligned16Bytes(rowTransLen * sizeof(T)) &&
               IsAligned16Bytes(splitTransLen * sizeof(T));
    }

    __aicore__ inline bool IsOddLargePureCopyLayout() const
    {
        return outerTile == B8_COMPACT_OUTER_TILE && chunkSplitNum == 1 && colChunkNum == splitNum &&
               (rowLength > 256 || (rowLength > 128 && (rowLength & 1U) != 0)) &&
               rowTransLen == B8_ADDR_COUNT * splitSize &&
               splitTransLen == B8_ADDR_COUNT * AlignUp32Bytes(splitSize * sizeof(T)) / sizeof(T);
    }

    __aicore__ inline void FirstVnchwB8Once(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool groupHighHalf,
                                            bool colHighHalf)
    {
        TransDataTo5HDParams params;
        params.dstHighHalf = groupHighHalf;
        params.srcHighHalf = colHighHalf;
        params.repeatTimes = static_cast<uint8_t>(rowTransLen / B8_ELEMS_PER_DB);
        params.dstRepStride = params.repeatTimes == 1 ? 0 : B8_ELEMS_PER_DB;
        params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;

        uint64_t dstList[B8_ADDR_COUNT];
        uint64_t srcList[B8_ADDR_COUNT];
        const uint32_t groupBase = groupHighHalf ? B8_ADDR_COUNT : 0;
        const uint32_t colBase = colHighHalf ? B8_ADDR_COUNT : 0;
        for (uint32_t j = 0; j < B8_ADDR_COUNT; ++j) {
            srcList[j] = reinterpret_cast<uint64_t>(src[(groupBase + j) * rowTransLen].GetPhyAddr());
            dstList[j] = reinterpret_cast<uint64_t>(dst[(colBase + j) * B8_ELEMS_PER_DB].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstList, srcList, params);
    }

    __aicore__ inline void FirstVnchwB8(const LocalTensor<T>& src, const LocalTensor<T>& dst)
    {
        FirstVnchwB8Once(src, dst, false, false);
        FirstVnchwB8Once(src, dst, true, false);
        FirstVnchwB8Once(src, dst, false, true);
        FirstVnchwB8Once(src, dst, true, true);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FirstVnchw256Once(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool oddGroup,
                                             bool pHighHalf)
    {
        SplitVFirstVnchw256Parity<T>(src, dst, rowTransLen, rowLength, oddGroup, pHighHalf, B8_ADDR_COUNT,
                                     B8_ELEMS_PER_DB);
    }

    __aicore__ inline void FirstVnchw256(const LocalTensor<T>& src, const LocalTensor<T>& dst)
    {
        FirstVnchw256Once(src, dst, false, false);
        FirstVnchw256Once(src, dst, false, true);
        FirstVnchw256Once(src, dst, true, false);
        FirstVnchw256Once(src, dst, true, true);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FirstVnchw512ParityOnce(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool oddGroup,
                                                   bool pHighHalf, uint32_t groupTransLen)
    {
        const bool halfShifted = !IsAligned32Bytes(B8_FULL_ROW_512_GROUP_ROWS * groupTransLen * sizeof(T));
        SplitVFirstVnchw512Parity<T>(src, dst, rowTransLen, groupTransLen, B8_ADDR_COUNT, oddGroup, pHighHalf,
                                     halfShifted, B8_ADDR_COUNT, B8_ELEMS_PER_DB);
    }

    __aicore__ inline void FirstVnchw512Parity(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                               uint32_t groupTransLen)
    {
        FirstVnchw512ParityOnce(src, dst, false, false, groupTransLen);
        FirstVnchw512ParityOnce(src, dst, false, true, groupTransLen);
        FirstVnchw512ParityOnce(src, dst, true, false, groupTransLen);
        FirstVnchw512ParityOnce(src, dst, true, true, groupTransLen);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SecondVnchwB8Once(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool groupHighHalf,
                                             bool pHighHalf)
    {
        TransDataTo5HDParams params;
        params.dstHighHalf = pHighHalf;
        params.srcHighHalf = groupHighHalf;
        params.repeatTimes = static_cast<uint8_t>(splitTransLen / B8_ELEMS_PER_DB);
        params.dstRepStride = params.repeatTimes == 1 ? 0 : 1;
        params.srcRepStride = params.repeatTimes == 1 ? 0 : B8_ELEMS_PER_DB;

        uint64_t dstList[B8_ADDR_COUNT];
        uint64_t srcList[B8_ADDR_COUNT];
        const uint32_t groupBase = groupHighHalf ? B8_ADDR_COUNT : 0;
        const uint32_t pBase = pHighHalf ? B8_ADDR_COUNT : 0;
        for (uint32_t j = 0; j < B8_ADDR_COUNT; ++j) {
            srcList[j] = reinterpret_cast<uint64_t>(src[(pBase + j) * B8_ELEMS_PER_DB].GetPhyAddr());
            dstList[j] = reinterpret_cast<uint64_t>(dst[(groupBase + j) * splitTransLen].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstList, srcList, params);
    }

    __aicore__ inline void SecondVnchwB8(const LocalTensor<T>& src, const LocalTensor<T>& dst)
    {
        SecondVnchwB8Once(src, dst, false, false);
        SecondVnchwB8Once(src, dst, true, false);
        SecondVnchwB8Once(src, dst, false, true);
        SecondVnchwB8Once(src, dst, true, true);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SecondVnchw256Once(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool oddGroup,
                                              bool pHighHalf, bool halfShifted, uint8_t repeatTimes,
                                              uint32_t curSplitTransLen)
    {
        SplitVSecondParityVnchw<T>(src, dst, curSplitTransLen, oddGroup, pHighHalf, halfShifted, repeatTimes,
                                   B8_ADDR_COUNT, B8_ELEMS_PER_DB);
    }

    __aicore__ inline void SecondVnchw256(const LocalTensor<T>& src, const LocalTensor<T>& dst, uint32_t curSplitSize)
    {
        const uint32_t curSplitTransLen = B8_FULL_ROW_256_GROUP_ROWS * curSplitSize;
        const bool halfShifted = curSplitSize % 4 == 2;
        const uint8_t lowRepeat = static_cast<uint8_t>((curSplitSize + 3) / 4);
        const uint8_t highRepeat = static_cast<uint8_t>(curSplitSize / 4);
        SecondVnchw256Once(src, dst, false, false, halfShifted, lowRepeat, curSplitTransLen);
        SecondVnchw256Once(src, dst, false, true, halfShifted, highRepeat, curSplitTransLen);
        SecondVnchw256Once(src, dst, true, false, halfShifted, lowRepeat, curSplitTransLen);
        SecondVnchw256Once(src, dst, true, true, halfShifted, highRepeat, curSplitTransLen);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SecondVnchw512ParityOnce(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool oddGroup,
                                                    bool pHighHalf, uint8_t repeatTimes, bool halfShifted,
                                                    uint32_t curSplitTransLen)
    {
        SplitVSecondParityVnchw<T>(src, dst, curSplitTransLen, oddGroup, pHighHalf, halfShifted, repeatTimes,
                                   B8_ADDR_COUNT, B8_ELEMS_PER_DB);
    }

    __aicore__ inline void SecondVnchw512Parity(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                                uint32_t curSplitSize)
    {
        const uint32_t curSplitTransLen = GetSecondVnchw512Pitch(curSplitSize);
        const bool halfShifted = !IsAligned32Bytes(B8_ADDR_COUNT * curSplitSize * sizeof(T));
        const uint8_t lowRepeat = static_cast<uint8_t>((curSplitSize + 1) / 2);
        const uint8_t highRepeat = static_cast<uint8_t>(curSplitSize / 2);
        SecondVnchw512ParityOnce(src, dst, false, false, lowRepeat, halfShifted, curSplitTransLen);
        SecondVnchw512ParityOnce(src, dst, false, true, highRepeat, halfShifted, curSplitTransLen);
        SecondVnchw512ParityOnce(src, dst, true, false, lowRepeat, halfShifted, curSplitTransLen);
        SecondVnchw512ParityOnce(src, dst, true, true, highRepeat, halfShifted, curSplitTransLen);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WaitMte2ToV() { SplitVSync<HardEvent::MTE2_V>(); }

    __aicore__ inline void WaitVToMte2() { SplitVSync<HardEvent::V_MTE2>(); }

    __aicore__ inline void WaitVToMte3() { SplitVSync<HardEvent::V_MTE3>(); }

    __aicore__ inline void WaitMte3ToV() { SplitVSync<HardEvent::MTE3_V>(); }

    __aicore__ inline LocalTensor<T> CopyInFullRowRaw(uint64_t outerBase, uint32_t outerReal)
    {
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        WaitVToMte2();
        const uint32_t copyElems = outerReal * rowLength;
        const uint32_t copyBytes = copyElems * sizeof(T);
        if (IsAligned32Bytes(copyBytes)) {
            DataCopy(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyElems);
        } else {
            DataCopyExtParams copyInParams = {1, copyBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyInParams, padParams);
        }
        inQueue.EnQue<T>(inputLocal);
        LocalTensor<T> firstInputRaw = inQueue.DeQue<T>();
        WaitMte2ToV();
        return firstInputRaw;
    }

    __aicore__ inline LocalTensor<T> CopyInTile(uint64_t outerBase, uint32_t outerReal)
    {
        const uint32_t compactGroupElems = B8_ADDR_COUNT * rowLength;
        if (rowTransLen == compactGroupElems) {
            return CopyInFullRowRaw(outerBase, outerReal);
        }
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        WaitVToMte2();
        const uint32_t fullGroupCount = outerReal / B8_ADDR_COUNT;
        const uint32_t tailRows = outerReal % B8_ADDR_COUNT;
        if (fullGroupCount != 0) {
            DataCopyExtParams copyInParams = {static_cast<uint16_t>(fullGroupCount),
                                              static_cast<uint32_t>(compactGroupElems * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {true, 0, static_cast<uint8_t>(rowTransLen - compactGroupElems), 0};
            DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyInParams, padParams);
        }
        if (tailRows != 0) {
            const uint32_t tailBytes = tailRows * rowLength * sizeof(T);
            DataCopyExtParams copyInParams = {1, tailBytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams = {true, 0, static_cast<uint8_t>(AlignUp32Bytes(tailBytes) - tailBytes),
                                                 0};
            DataCopyPad(inputLocal[fullGroupCount * rowTransLen],
                        inputGm[(outerBase + static_cast<uint64_t>(fullGroupCount) * B8_ADDR_COUNT) * rowLength],
                        copyInParams, padParams);
        }
        inQueue.EnQue<T>(inputLocal);

        LocalTensor<T> firstInputRaw = inQueue.DeQue<T>();
        WaitMte2ToV();
        return firstInputRaw;
    }

    __aicore__ inline LocalTensor<T> CopyInChunk(uint64_t outerBase, uint32_t outerReal, uint32_t colBase,
                                                 uint32_t chunkCols)
    {
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        WaitVToMte2();

        const uint32_t chunkRowPitch = rowTransLen / B8_ADDR_COUNT;
        const uint32_t chunkBytes = chunkCols * sizeof(T);
        const uint32_t alignedChunkBytes = AlignUp32Bytes(chunkBytes);
        const uint32_t chunkRowPitchBytes = chunkRowPitch * sizeof(T);
        const uint32_t rightPadding = (alignedChunkBytes - chunkBytes) / sizeof(T);
        DataCopyExtParams copyInParams = {
            static_cast<uint16_t>(outerReal), chunkBytes, static_cast<uint32_t>((rowLength - chunkCols) * sizeof(T)),
            static_cast<uint32_t>((chunkRowPitchBytes - alignedChunkBytes) / B8_ELEMS_PER_DB), 0};
        DataCopyPadExtParams<T> padParams = {rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), 0};
        DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength) + colBase], copyInParams,
                    padParams);
        inQueue.EnQue<T>(inputLocal);

        LocalTensor<T> firstInputRaw = inQueue.DeQue<T>();
        WaitMte2ToV();
        return firstInputRaw;
    }

    __aicore__ inline LocalTensor<T> CopyInPureCopy(uint64_t outerBase, uint32_t outerReal, uint32_t colBase,
                                                    uint32_t curSplitSize)
    {
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        WaitVToMte2();

        const uint32_t splitBytes = curSplitSize * sizeof(T);
        DataCopyExtParams copyInParams = {static_cast<uint16_t>(outerReal), splitBytes,
                                          static_cast<uint32_t>((rowLength - curSplitSize) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength) + colBase], copyInParams,
                    padParams);
        inQueue.EnQue<T>(inputLocal);

        LocalTensor<T> firstInputRaw = inQueue.DeQue<T>();
        WaitMte2ToV();
        return firstInputRaw;
    }

    __aicore__ inline LocalTensor<T> CopyInFullRow256(uint64_t outerBase, uint32_t outerReal)
    {
        return CopyInFullRowRaw(outerBase, outerReal);
    }

    __aicore__ inline void CompactSplitSegment(LocalTensor<T> seg, uint32_t localSplitIndex, uint32_t curSplitSize,
                                               uint32_t srcStride, uint32_t dstStride)
    {
        DataCopyParams segCopyParams;
        segCopyParams.blockCount = B8_ADDR_COUNT;
        segCopyParams.blockLen = static_cast<uint16_t>(curSplitSize);
        segCopyParams.srcStride = static_cast<uint16_t>(srcStride);
        segCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        WaitVToMte2();
        DataCopy(seg, transTensor[localSplitIndex * splitSize * B8_ELEMS_PER_DB], segCopyParams);
        WaitMte2ToV();
    }

    __aicore__ inline void CompactSplitSegment256(LocalTensor<T> seg, uint32_t splitIndex)
    {
        const uint32_t curSplitSize = GetSplitSize(splitIndex);
        DataCopyParams segCopyParams;
        segCopyParams.blockCount = B8_FULL_ROW_256_GROUP_ROWS;
        segCopyParams.blockLen = static_cast<uint16_t>(curSplitSize);
        segCopyParams.srcStride = static_cast<uint16_t>(rowLength - curSplitSize);
        segCopyParams.dstStride = 0;
        WaitVToMte2();
        DataCopy(seg, transTensor[splitIndex * splitSize * B8_ELEMS_PER_DB], segCopyParams);
        WaitMte2ToV();
    }

    __aicore__ inline uint32_t GetGroupValidRows(uint32_t groupIdx, uint32_t outerReal) const
    {
        const uint32_t groupRowBase = groupIdx * B8_ADDR_COUNT;
        if (outerReal <= groupRowBase) {
            return 0;
        }
        const uint32_t remainRows = outerReal - groupRowBase;
        return remainRows > B8_ADDR_COUNT ? B8_ADDR_COUNT : remainRows;
    }

    __aicore__ inline void SecondVnchwB8AndCopyOut(const LocalTensor<T>& seg, uint32_t outputIndex, uint64_t outerBase,
                                                   uint32_t outerReal)
    {
        const uint32_t curSplitSize = GetSplitSize(outputIndex);
        LocalTensor<T> outputLocalRaw = outQueue.AllocTensor<T>();
        SecondVnchwB8(seg, outputLocalRaw);
        outQueue.EnQue<T>(outputLocalRaw);

        LocalTensor<T> resultLocal = outQueue.DeQue<T>();
        WaitVToMte3();
        outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex),
                                 outerLength * static_cast<uint64_t>(curSplitSize));
        for (uint32_t groupIdx = 0; groupIdx < B8_ELEMS_PER_DB; ++groupIdx) {
            const uint32_t validRows = GetGroupValidRows(groupIdx, outerReal);
            if (validRows == 0) {
                break;
            }
            const uint32_t copyElems = validRows * curSplitSize;
            const uint64_t gmOffset = (outerBase + static_cast<uint64_t>(groupIdx) * B8_ADDR_COUNT) * curSplitSize;
            const uint32_t localOffset = groupIdx * splitTransLen;
            CopyOutFlatResult(resultLocal[localOffset], gmOffset, copyElems);
        }
        WaitMte3ToV();
        outQueue.FreeTensor(resultLocal);
    }

    __aicore__ inline void SecondVnchwB8AndCopyOutChunk(const LocalTensor<T>& seg, uint32_t outputIndex,
                                                        uint64_t outerBase, uint32_t outerReal)
    {
        SecondVnchwB8AndCopyOut(seg, outputIndex, outerBase, outerReal);
    }

    __aicore__ inline void CopyOutFlatResult(const LocalTensor<T>& resultLocal, uint64_t gmOffset, uint32_t copyElems)
    {
        const uint32_t copyBytes = copyElems * sizeof(T);
        if (IsAligned32Bytes(copyBytes)) {
            DataCopy(outputGm[gmOffset], resultLocal, copyElems);
        } else {
            DataCopyExtParams copyOutParams = {1, copyBytes, 0, 0, 0};
            DataCopyPad(outputGm[gmOffset], resultLocal, copyOutParams);
        }
    }

    __aicore__ inline void SecondVnchw256AndCopyOut(const LocalTensor<T>& seg, uint32_t outputIndex, uint64_t outerBase,
                                                    uint32_t outerReal)
    {
        const uint32_t curSplitSize = GetSplitSize(outputIndex);
        LocalTensor<T> outputLocalRaw = outQueue.AllocTensor<T>();
        SecondVnchw256(seg, outputLocalRaw, curSplitSize);
        outQueue.EnQue<T>(outputLocalRaw);

        LocalTensor<T> resultLocal = outQueue.DeQue<T>();
        WaitVToMte3();
        outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex),
                                 outerLength * static_cast<uint64_t>(curSplitSize));
        if (curSplitSize == splitSize) {
            const uint32_t copyElems = outerReal * splitSize;
            const uint64_t gmOffset = outerBase * static_cast<uint64_t>(splitSize);
            CopyOutFlatResult(resultLocal, gmOffset, copyElems);
        } else {
            const uint32_t curSplitTransLen = B8_FULL_ROW_256_GROUP_ROWS * curSplitSize;
            CopyOutGroupedResult(resultLocal, curSplitSize, outerBase, outerReal, B8_FULL_ROW_256_GROUP_ROWS,
                                 curSplitTransLen);
        }
        WaitMte3ToV();
        outQueue.FreeTensor(resultLocal);
    }

    __aicore__ inline void SecondVnchw512ParityAndCopyOut(const LocalTensor<T>& seg, uint32_t outputIndex,
                                                          uint64_t outerBase, uint32_t outerReal)
    {
        const uint32_t curSplitSize = GetSplitSize(outputIndex);
        LocalTensor<T> outputLocalRaw = outQueue.AllocTensor<T>();
        SecondVnchw512Parity(seg, outputLocalRaw, curSplitSize);
        outQueue.EnQue<T>(outputLocalRaw);

        LocalTensor<T> resultLocal = outQueue.DeQue<T>();
        WaitVToMte3();
        outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex),
                                 outerLength * static_cast<uint64_t>(curSplitSize));
        const uint32_t copyElems = outerReal * curSplitSize;
        const uint64_t gmOffset = outerBase * static_cast<uint64_t>(curSplitSize);
        CopyOutFlatResult(resultLocal, gmOffset, copyElems);
        WaitMte3ToV();
        outQueue.FreeTensor(resultLocal);
    }

    __aicore__ inline void CopyOutGroupedResult(const LocalTensor<T>& resultLocal, uint32_t curSplitSize,
                                                uint64_t outerBase, uint32_t outerReal, uint32_t groupRows,
                                                uint32_t localGroupStride)
    {
        for (uint32_t groupIdx = 0; groupIdx * groupRows < outerReal; ++groupIdx) {
            const uint32_t groupRowBase = groupIdx * groupRows;
            const uint32_t remainRows = outerReal - groupRowBase;
            const uint32_t validRows = remainRows > groupRows ? groupRows : remainRows;
            const uint32_t copyElems = validRows * curSplitSize;
            const uint64_t gmOffset = (outerBase + static_cast<uint64_t>(groupRowBase)) * curSplitSize;
            const uint32_t localOffset = groupIdx * localGroupStride;
            CopyOutFlatResult(resultLocal[localOffset], gmOffset, copyElems);
        }
    }

    __aicore__ inline void CopyOutPureCopy(const LocalTensor<T>& inputLocal, uint32_t outputIndex, uint64_t outerBase,
                                           uint32_t outerReal)
    {
        const uint32_t curSplitSize = GetSplitSize(outputIndex);
        WaitVToMte3();
        outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex),
                                 outerLength * static_cast<uint64_t>(curSplitSize));

        const uint64_t gmOffset = outerBase * static_cast<uint64_t>(curSplitSize);
        const uint32_t copyElems = outerReal * curSplitSize;
        const uint32_t copyBytes = copyElems * sizeof(T);
        const uint32_t splitPitch = splitTransLen / B8_ADDR_COUNT;
        if (splitPitch == curSplitSize && IsAligned32Bytes(copyBytes)) {
            DataCopy(outputGm[gmOffset], inputLocal, copyElems);
        } else {
            DataCopyExtParams copyOutParams = {static_cast<uint16_t>(outerReal),
                                               static_cast<uint32_t>(curSplitSize * sizeof(T)),
                                               static_cast<uint32_t>((splitPitch - curSplitSize) * sizeof(T)), 0, 0};
            DataCopyPad(outputGm[gmOffset], inputLocal, copyOutParams);
        }
        WaitMte3ToV();
    }

    __aicore__ inline void ProcessTile(uint64_t outerBase, uint32_t outerReal)
    {
        if (IsFullRow512ParityLayout()) {
            ProcessFullRow512Parity(outerBase, outerReal);
            return;
        }

        LocalTensor<T> inputLocal = CopyInTile(outerBase, outerReal);
        FirstVnchwB8(inputLocal, transTensor);

        for (uint32_t g = 0; g < splitNum; ++g) {
            const uint32_t curSplitSize = GetSplitSize(g);
            CompactSplitSegment(inputLocal, g, curSplitSize, rowLength - curSplitSize, 0);
            SecondVnchwB8AndCopyOut(inputLocal, g, outerBase, outerReal);
        }
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessFullRow512Parity(uint64_t outerBase, uint32_t outerReal)
    {
        LocalTensor<T> inputLocal = CopyInTile(outerBase, outerReal);
        FirstVnchw512Parity(inputLocal, transTensor, rowLength);

        for (uint32_t g = 0; g < splitNum; ++g) {
            const uint32_t curSplitSize = GetSplitSize(g);
            CompactSplitSegment(inputLocal, g, curSplitSize, rowLength - curSplitSize, 0);
            SecondVnchw512ParityAndCopyOut(inputLocal, g, outerBase, outerReal);
        }
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessFullRow256(uint64_t outerBase, uint32_t outerReal)
    {
        LocalTensor<T> inputLocal = CopyInFullRow256(outerBase, outerReal);
        FirstVnchw256(inputLocal, transTensor);

        for (uint32_t g = 0; g < splitNum; ++g) {
            CompactSplitSegment256(inputLocal, g);
            SecondVnchw256AndCopyOut(inputLocal, g, outerBase, outerReal);
        }
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessPureCopyTask(uint64_t taskIdx)
    {
        const uint64_t outerTileIdx = taskIdx / colChunkNum;
        const uint32_t splitIndex = static_cast<uint32_t>(taskIdx % colChunkNum);
        const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
        const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
        if (splitIndex >= splitNum) {
            return;
        }

        const uint32_t colBase = splitIndex * splitSize;
        const uint32_t curSplitSize = GetSplitSize(splitIndex);
        LocalTensor<T> inputLocal = CopyInPureCopy(outerBase, outerReal, colBase, curSplitSize);
        CopyOutPureCopy(inputLocal, splitIndex, outerBase, outerReal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ProcessChunkTask(uint64_t taskIdx)
    {
        const uint64_t outerTileIdx = taskIdx / colChunkNum;
        const uint32_t colChunkIdx = static_cast<uint32_t>(taskIdx % colChunkNum);
        const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
        const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;

        const uint32_t splitBase = colChunkIdx * chunkSplitNum;
        if (splitBase >= splitNum) {
            return;
        }
        const uint32_t remainSplits = splitNum - splitBase;
        const uint32_t curChunkSplitNum = remainSplits > chunkSplitNum ? chunkSplitNum : remainSplits;
        const uint32_t colBase = splitBase * splitSize;
        const uint32_t lastSplitInChunk = splitBase + curChunkSplitNum - 1;
        const uint32_t chunkCols = (curChunkSplitNum - 1) * splitSize + GetSplitSize(lastSplitInChunk);
        const uint32_t chunkRowPitch = rowTransLen / B8_ADDR_COUNT;

        LocalTensor<T> inputLocal = CopyInChunk(outerBase, outerReal, colBase, chunkCols);
        if (IsChunk512ParityLayout() && chunkCols == chunkRowPitch) {
            FirstVnchw512Parity(inputLocal, transTensor, chunkRowPitch);
            for (uint32_t localSplit = 0; localSplit < curChunkSplitNum; ++localSplit) {
                const uint32_t curSplitSize = GetSplitSize(splitBase + localSplit);
                CompactSplitSegment(inputLocal, localSplit, curSplitSize, chunkRowPitch - curSplitSize, 0);
                SecondVnchw512ParityAndCopyOut(inputLocal, splitBase + localSplit, outerBase, outerReal);
            }
            inQueue.FreeTensor(inputLocal);
            return;
        }

        FirstVnchwB8(inputLocal, transTensor);

        for (uint32_t localSplit = 0; localSplit < curChunkSplitNum; ++localSplit) {
            const uint32_t curSplitSize = GetSplitSize(splitBase + localSplit);
            CompactSplitSegment(inputLocal, localSplit, curSplitSize, chunkRowPitch - curSplitSize, 0);
            SecondVnchwB8AndCopyOutChunk(inputLocal, splitBase + localSplit, outerBase, outerReal);
        }
        inQueue.FreeTensor(inputLocal);
    }

    TPipe* pipe = nullptr;
    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> transBuffer;
    LocalTensor<T> transTensor;

    uint64_t totalLength = 0;
    uint64_t outerLength = 0;
    uint32_t rowLength = 0;
    uint32_t splitSize = 0;
    uint32_t tailSplitSize = 0;
    uint32_t splitNum = 0;
    uint32_t outerTile = 0;
    uint32_t outerTileNum = 0;
    uint32_t outerTail = 0;
    uint64_t tileNum = 0;
    uint64_t loopOff = 0;
    uint32_t rowTransLen = 0;
    uint32_t splitTransLen = 0;
    uint32_t chunkSplitNum = 0;
    uint32_t colChunkNum = 0;
};

#endif // OP_KERNEL_SPLIT_V_SAME_LEN_COMPACT_8BIT_H_
