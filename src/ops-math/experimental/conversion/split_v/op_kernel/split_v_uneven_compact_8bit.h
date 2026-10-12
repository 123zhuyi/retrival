/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_UNEVEN_COMPACT_8BIT_H_
#define OP_KERNEL_SPLIT_V_UNEVEN_COMPACT_8BIT_H_
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "split_v_kernel_common.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename T>
class SplitVUnevenCompact8Bit : private SplitVUnevenCompactTilingState {
public:
    __aicore__ inline SplitVUnevenCompact8Bit() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingDataUnevenCompact* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    static constexpr uint32_t BUFFER_NUM = 1;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t B8_ADDR_COUNT = 16;
    static constexpr uint32_t B8_ELEMS_PER_DB = 32;
    static constexpr uint32_t PURE_COPY_COL_LIMIT = 512;
    static constexpr uint32_t OUTER_TILE_512 = 512;
    static constexpr uint32_t OUTER_TILE_256 = 256;
    static constexpr uint32_t GROUP_ROWS_512 = 16;
    static constexpr uint32_t GROUP_ROWS_256 = 8;

    static constexpr uint32_t MODE_PURE_COPY = 0;
    static constexpr uint32_t MODE_FULL_512_PARITY = 1;
    static constexpr uint32_t MODE_FULL_256_VIRTUAL = 2;
    static constexpr uint32_t MODE_CHUNK_512_PARITY = 3;

    using Base = SplitVUnevenCompactTilingState;
    SPLIT_V_USE_UNEVEN_COMPACT_TILING_STATE(Base);

    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> transBuf;
    LocalTensor<T> transTensor;

    uint32_t rowTransLen = 0;
    uint32_t splitTransLen = 0;
    uint32_t virtualSplitSize = 0;
    uint32_t virtualSplitNum = 0;
    uint32_t pureSplitPitch = 0;

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const;
    __aicore__ inline bool IsAligned32Bytes(uint32_t bytes) const;
    __aicore__ inline bool IsAligned16Bytes(uint32_t bytes) const;
    __aicore__ inline uint32_t GetInputStageBytes() const;
    __aicore__ inline uint32_t GetOutputStageBytes() const;
    __aicore__ inline uint32_t GetTransStageBytes() const;
    __aicore__ inline bool IsValidTiling() const;

    __aicore__ inline void WaitMte2ToV();
    __aicore__ inline void WaitVToMte2();
    __aicore__ inline void WaitVToMte3();
    __aicore__ inline void WaitMte3ToV();

    __aicore__ inline LocalTensor<T> CopyInFullRow(uint64_t outerBase, uint32_t outerReal);
    __aicore__ inline LocalTensor<T> CopyInChunk(uint64_t outerBase, uint32_t outerReal, uint32_t colBase,
                                                 uint32_t chunkCols);
    __aicore__ inline LocalTensor<T> CopyInPure(uint64_t outerBase, uint32_t outerReal, uint32_t splitIndex,
                                                uint32_t colOff, uint32_t copyLen);

    __aicore__ inline void FirstVnchw512ParityOnce(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool oddGroup,
                                                   bool pHighHalf, uint32_t groupTransLen);
    __aicore__ inline void FirstVnchw512Parity(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                               uint32_t groupTransLen);
    __aicore__ inline void FirstVnchw256Once(const LocalTensor<T>& src, const LocalTensor<T>& dst, bool oddGroup,
                                             bool pHighHalf);
    __aicore__ inline void FirstVnchw256(const LocalTensor<T>& src, const LocalTensor<T>& dst);

    __aicore__ inline void CompactSegment512(LocalTensor<T> seg, uint32_t colStart, uint32_t splitLen,
                                             uint32_t sourceRowLen);
    __aicore__ inline void CompactSegment256(LocalTensor<T> seg, uint32_t colStart, uint32_t splitLen);

    __aicore__ inline void SecondVnchw512ParityOnce(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                                    uint32_t curSplitTransLen, bool oddGroup, bool pHighHalf,
                                                    uint8_t repeatTimes);
    __aicore__ inline void SecondVnchw512Parity(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                                uint32_t splitLen);
    __aicore__ inline void SecondVnchw256Once(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                              uint32_t curSplitTransLen, bool oddGroup, bool pHighHalf,
                                              bool halfShifted, uint8_t repeatTimes);
    __aicore__ inline void SecondVnchw256(const LocalTensor<T>& src, const LocalTensor<T>& dst, uint32_t splitLen);

    __aicore__ inline void CopyOutContiguous(const LocalTensor<T>& resultLocal, uint32_t outputIndex,
                                             uint64_t outerBase, uint32_t outerReal, uint32_t splitLen);
    __aicore__ inline void CopyOutOverlap(const LocalTensor<T>& resultLocal, uint32_t outputIndex, uint64_t outerBase,
                                          uint32_t outerReal, uint32_t actualLen, uint32_t localLen, uint32_t srcColOff,
                                          uint32_t dstColOff, uint32_t copyLen);
    __aicore__ inline void CopyOutPure(const LocalTensor<T>& inputLocal, uint32_t outputIndex, uint64_t outerBase,
                                       uint32_t outerReal, uint32_t splitLen, uint32_t colOff, uint32_t copyLen);

    __aicore__ inline void ProcessFullRow512(uint64_t outerBase, uint32_t outerReal);
    __aicore__ inline void ProcessFullRow256Virtual(uint64_t outerBase, uint32_t outerReal);
    __aicore__ inline void ProcessChunk512(uint64_t taskIdx);
    __aicore__ inline void ProcessPureTile(uint64_t outerBase, uint32_t outerReal);
};

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::Init(GM_ADDR input, GM_ADDR output,
                                                        SplitVTilingDataUnevenCompact* tiling, TPipe* pipe)
{
    const uint32_t blockIdx = GetBlockIdx();
    outList.Init((__gm__ void*)output);
    LoadUnevenCompactTiling(tiling, blockIdx);
    rowTransLen = tiling->rowTransLen;
    splitTransLen = tiling->splitTransLen;
    virtualSplitSize = tiling->virtualSplitSize;
    virtualSplitNum = tiling->virtualSplitNum;
    pureSplitPitch = AlignUp(maxSplitSize * sizeof(T), BLOCK_SIZE) / sizeof(T);

    inputGm.SetGlobalBuffer((__gm__ T*)input, totalLength);
    pipe->InitBuffer(inQueue, BUFFER_NUM, GetInputStageBytes());
    pipe->InitBuffer(outQueue, BUFFER_NUM, GetOutputStageBytes());
    pipe->InitBuffer(transBuf, GetTransStageBytes());
    transTensor = transBuf.Get<T>();
}

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact8Bit<T>::AlignUp(uint32_t value, uint32_t align) const
{
    return SplitVAlignUp(value, align);
}

template <typename T>
__aicore__ inline bool SplitVUnevenCompact8Bit<T>::IsAligned32Bytes(uint32_t bytes) const
{
    return SplitVIsAlignedBytes(bytes, BLOCK_SIZE);
}

template <typename T>
__aicore__ inline bool SplitVUnevenCompact8Bit<T>::IsAligned16Bytes(uint32_t bytes) const
{
    return SplitVIsAlignedBytes(bytes, B8_ADDR_COUNT);
}

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact8Bit<T>::GetInputStageBytes() const
{
    if (mode == MODE_PURE_COPY) {
        return outerTile * pureSplitPitch * sizeof(T);
    }
    return B8_ELEMS_PER_DB * rowTransLen * sizeof(T);
}

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact8Bit<T>::GetOutputStageBytes() const
{
    if (mode == MODE_PURE_COPY) {
        return outerTile * pureSplitPitch * sizeof(T);
    }
    return B8_ELEMS_PER_DB * splitTransLen * sizeof(T);
}

template <typename T>
__aicore__ inline uint32_t SplitVUnevenCompact8Bit<T>::GetTransStageBytes() const
{
    if (mode == MODE_PURE_COPY) {
        return B8_ELEMS_PER_DB * sizeof(T);
    }
    return B8_ELEMS_PER_DB * rowTransLen * sizeof(T);
}

template <typename T>
__aicore__ inline bool SplitVUnevenCompact8Bit<T>::IsValidTiling() const
{
    if (outerLength == 0 || rowLength == 0 || splitNum == 0 || maxSplitSize == 0 || outerTile == 0 ||
        outerTileNum == 0 || outerTail == 0) {
        return false;
    }
    if (mode == MODE_PURE_COPY) {
        return pureSplitPitch >= maxSplitSize && IsAligned32Bytes(pureSplitPitch * sizeof(T));
    }
    if (mode == MODE_FULL_512_PARITY) {
        return outerTile == OUTER_TILE_512 && colChunkNum == 1 && rowTransLen == GROUP_ROWS_512 * rowLength &&
               splitTransLen == GROUP_ROWS_512 * maxSplitSize && IsAligned16Bytes(rowTransLen * sizeof(T)) &&
               IsAligned16Bytes(splitTransLen * sizeof(T)) && (rowLength + 1) / 2 <= UINT8_MAX &&
               (maxSplitSize + 1) / 2 <= UINT8_MAX;
    }
    if (mode == MODE_FULL_256_VIRTUAL) {
        return outerTile == OUTER_TILE_256 && rowLength > 128 && rowLength % 4 == 0 && virtualSplitSize != 0 &&
               virtualSplitNum != 0 && virtualSplitSize % 2 == 0 && virtualSplitNum * virtualSplitSize == rowLength &&
               rowTransLen == GROUP_ROWS_256 * rowLength && splitTransLen == GROUP_ROWS_256 * virtualSplitSize &&
               rowLength / 4 <= UINT8_MAX && (virtualSplitSize + 3) / 4 <= UINT8_MAX;
    }
    return mode == MODE_CHUNK_512_PARITY && outerTile == OUTER_TILE_512 && colChunkSize != 0 && colChunkNum != 0 &&
           rowTransLen == GROUP_ROWS_512 * colChunkSize && splitTransLen == GROUP_ROWS_512 * maxSplitSize &&
           (colChunkSize + 1) / 2 <= UINT8_MAX && (maxSplitSize + 1) / 2 <= UINT8_MAX;
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::WaitMte2ToV()
{
    SplitVSync<HardEvent::MTE2_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::WaitVToMte2()
{
    SplitVSync<HardEvent::V_MTE2>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::WaitVToMte3()
{
    SplitVSync<HardEvent::V_MTE3>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::WaitMte3ToV()
{
    SplitVSync<HardEvent::MTE3_V>();
}

template <typename T>
__aicore__ inline LocalTensor<T> SplitVUnevenCompact8Bit<T>::CopyInFullRow(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
    WaitVToMte2();
    const uint32_t copyElems = outerReal * rowLength;
    const uint32_t copyBytes = copyElems * sizeof(T);
    if (IsAligned32Bytes(copyBytes)) {
        DataCopy(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyElems);
    } else {
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyParams, padParams);
    }
    inQueue.EnQue(inputLocal);
    LocalTensor<T> raw = inQueue.DeQue<T>();
    WaitMte2ToV();
    return raw;
}

template <typename T>
__aicore__ inline LocalTensor<T> SplitVUnevenCompact8Bit<T>::CopyInChunk(uint64_t outerBase, uint32_t outerReal,
                                                                         uint32_t colBase, uint32_t chunkCols)
{
    LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
    WaitVToMte2();
    const uint32_t chunkRowPitch = rowTransLen / GROUP_ROWS_512;
    const uint32_t chunkBytes = chunkCols * sizeof(T);
    const uint32_t alignedChunkBytes = AlignUp(chunkBytes, BLOCK_SIZE);
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(outerReal), chunkBytes, static_cast<uint32_t>((rowLength - chunkCols) * sizeof(T)),
        static_cast<uint32_t>((chunkRowPitch * sizeof(T) - alignedChunkBytes) / BLOCK_SIZE), 0};
    DataCopyPadExtParams<T> padParams = {alignedChunkBytes != chunkBytes, 0,
                                         static_cast<uint8_t>((alignedChunkBytes - chunkBytes) / sizeof(T)), 0};
    DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength) + colBase], copyParams, padParams);
    inQueue.EnQue(inputLocal);
    LocalTensor<T> raw = inQueue.DeQue<T>();
    WaitMte2ToV();
    return raw;
}

template <typename T>
__aicore__ inline LocalTensor<T> SplitVUnevenCompact8Bit<T>::CopyInPure(uint64_t outerBase, uint32_t outerReal,
                                                                        uint32_t splitIndex, uint32_t colOff,
                                                                        uint32_t copyLen)
{
    LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
    WaitVToMte2();
    const uint32_t copyPitch = AlignUp(copyLen * sizeof(T), BLOCK_SIZE) / sizeof(T);
    const uint32_t rightPadding = copyPitch - copyLen;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(outerReal), static_cast<uint32_t>(copyLen * sizeof(T)),
                                    static_cast<uint32_t>((rowLength - copyLen) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams = {rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), 0};
    DataCopyPad(inputLocal, inputGm[outerBase * static_cast<uint64_t>(rowLength) + splitStarts[splitIndex] + colOff],
                copyParams, padParams);
    inQueue.EnQue(inputLocal);
    LocalTensor<T> raw = inQueue.DeQue<T>();
    WaitMte2ToV();
    return raw;
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::FirstVnchw512ParityOnce(const LocalTensor<T>& src,
                                                                           const LocalTensor<T>& dst, bool oddGroup,
                                                                           bool pHighHalf, uint32_t groupTransLen)
{
    const bool halfShifted = !IsAligned32Bytes(GROUP_ROWS_512 * groupTransLen * sizeof(T));
    SplitVFirstVnchw512Parity<T>(src, dst, rowTransLen, groupTransLen, B8_ADDR_COUNT, oddGroup, pHighHalf, halfShifted,
                                 B8_ADDR_COUNT, B8_ELEMS_PER_DB);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::FirstVnchw512Parity(const LocalTensor<T>& src,
                                                                       const LocalTensor<T>& dst,
                                                                       uint32_t groupTransLen)
{
    FirstVnchw512ParityOnce(src, dst, false, false, groupTransLen);
    FirstVnchw512ParityOnce(src, dst, false, true, groupTransLen);
    FirstVnchw512ParityOnce(src, dst, true, false, groupTransLen);
    FirstVnchw512ParityOnce(src, dst, true, true, groupTransLen);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::FirstVnchw256Once(const LocalTensor<T>& src,
                                                                     const LocalTensor<T>& dst, bool oddGroup,
                                                                     bool pHighHalf)
{
    SplitVFirstVnchw256Parity<T>(src, dst, rowTransLen, rowLength, oddGroup, pHighHalf, B8_ADDR_COUNT, B8_ELEMS_PER_DB);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::FirstVnchw256(const LocalTensor<T>& src, const LocalTensor<T>& dst)
{
    FirstVnchw256Once(src, dst, false, false);
    FirstVnchw256Once(src, dst, false, true);
    FirstVnchw256Once(src, dst, true, false);
    FirstVnchw256Once(src, dst, true, true);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::CompactSegment512(LocalTensor<T> seg, uint32_t colStart,
                                                                     uint32_t splitLen, uint32_t sourceRowLen)
{
    DataCopyParams copyParams;
    copyParams.blockCount = B8_ADDR_COUNT;
    copyParams.blockLen = static_cast<uint16_t>(splitLen);
    copyParams.srcStride = static_cast<uint16_t>(sourceRowLen - splitLen);
    copyParams.dstStride = 0;
    WaitVToMte2();
    DataCopy(seg, transTensor[colStart * B8_ELEMS_PER_DB], copyParams);
    WaitMte2ToV();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::CompactSegment256(LocalTensor<T> seg, uint32_t colStart,
                                                                     uint32_t splitLen)
{
    DataCopyParams copyParams;
    copyParams.blockCount = GROUP_ROWS_256;
    copyParams.blockLen = static_cast<uint16_t>(splitLen);
    copyParams.srcStride = static_cast<uint16_t>(rowLength - splitLen);
    copyParams.dstStride = 0;
    WaitVToMte2();
    DataCopy(seg, transTensor[colStart * B8_ELEMS_PER_DB], copyParams);
    WaitMte2ToV();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::SecondVnchw512ParityOnce(const LocalTensor<T>& src,
                                                                            const LocalTensor<T>& dst,
                                                                            uint32_t curSplitTransLen, bool oddGroup,
                                                                            bool pHighHalf, uint8_t repeatTimes)
{
    const bool halfShifted = !IsAligned32Bytes(curSplitTransLen * sizeof(T));
    SplitVSecondParityVnchw<T>(src, dst, curSplitTransLen, oddGroup, pHighHalf, halfShifted, repeatTimes, B8_ADDR_COUNT,
                               B8_ELEMS_PER_DB);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::SecondVnchw512Parity(const LocalTensor<T>& src,
                                                                        const LocalTensor<T>& dst, uint32_t splitLen)
{
    const uint32_t curSplitTransLen = GROUP_ROWS_512 * splitLen;
    const uint8_t lowRepeat = static_cast<uint8_t>((splitLen + 1) / 2);
    const uint8_t highRepeat = static_cast<uint8_t>(splitLen / 2);
    SecondVnchw512ParityOnce(src, dst, curSplitTransLen, false, false, lowRepeat);
    SecondVnchw512ParityOnce(src, dst, curSplitTransLen, false, true, highRepeat);
    SecondVnchw512ParityOnce(src, dst, curSplitTransLen, true, false, lowRepeat);
    SecondVnchw512ParityOnce(src, dst, curSplitTransLen, true, true, highRepeat);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::SecondVnchw256Once(const LocalTensor<T>& src,
                                                                      const LocalTensor<T>& dst,
                                                                      uint32_t curSplitTransLen, bool oddGroup,
                                                                      bool pHighHalf, bool halfShifted,
                                                                      uint8_t repeatTimes)
{
    SplitVSecondParityVnchw<T>(src, dst, curSplitTransLen, oddGroup, pHighHalf, halfShifted, repeatTimes, B8_ADDR_COUNT,
                               B8_ELEMS_PER_DB);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::SecondVnchw256(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                                                  uint32_t splitLen)
{
    const uint32_t curSplitTransLen = GROUP_ROWS_256 * splitLen;
    const bool halfShifted = splitLen % 4 == 2;
    const uint8_t lowRepeat = static_cast<uint8_t>((splitLen + 3) / 4);
    const uint8_t highRepeat = static_cast<uint8_t>(splitLen / 4);
    SecondVnchw256Once(src, dst, curSplitTransLen, false, false, halfShifted, lowRepeat);
    SecondVnchw256Once(src, dst, curSplitTransLen, false, true, halfShifted, highRepeat);
    SecondVnchw256Once(src, dst, curSplitTransLen, true, false, halfShifted, lowRepeat);
    SecondVnchw256Once(src, dst, curSplitTransLen, true, true, halfShifted, highRepeat);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::CopyOutContiguous(const LocalTensor<T>& resultLocal,
                                                                     uint32_t outputIndex, uint64_t outerBase,
                                                                     uint32_t outerReal, uint32_t splitLen)
{
    WaitVToMte3();
    outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex), outerLength * static_cast<uint64_t>(splitLen));
    const uint32_t copyElems = outerReal * splitLen;
    const uint32_t copyBytes = copyElems * sizeof(T);
    const uint64_t gmOffset = outerBase * static_cast<uint64_t>(splitLen);
    if (IsAligned32Bytes(copyBytes)) {
        DataCopy(outputGm[gmOffset], resultLocal, copyElems);
    } else {
        DataCopyExtParams copyParams = {1, copyBytes, 0, 0, 0};
        DataCopyPad(outputGm[gmOffset], resultLocal, copyParams);
    }
    WaitMte3ToV();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::CopyOutOverlap(const LocalTensor<T>& resultLocal,
                                                                  uint32_t outputIndex, uint64_t outerBase,
                                                                  uint32_t outerReal, uint32_t actualLen,
                                                                  uint32_t localLen, uint32_t srcColOff,
                                                                  uint32_t dstColOff, uint32_t copyLen)
{
    WaitVToMte3();
    outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex), outerLength * static_cast<uint64_t>(actualLen));
    DataCopyExtParams copyParams = {static_cast<uint16_t>(outerReal), static_cast<uint32_t>(copyLen * sizeof(T)),
                                    static_cast<uint32_t>((localLen - copyLen) * sizeof(T)),
                                    static_cast<uint32_t>((actualLen - copyLen) * sizeof(T)), 0};
    const uint64_t gmOffset = outerBase * static_cast<uint64_t>(actualLen) + dstColOff;
    DataCopyPad(outputGm[gmOffset], resultLocal[srcColOff], copyParams);
    WaitMte3ToV();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::CopyOutPure(const LocalTensor<T>& inputLocal, uint32_t outputIndex,
                                                               uint64_t outerBase, uint32_t outerReal,
                                                               uint32_t splitLen, uint32_t colOff, uint32_t copyLen)
{
    WaitVToMte3();
    outputGm.SetGlobalBuffer(outList.GetDataPtr<__gm__ T>(outputIndex), outerLength * static_cast<uint64_t>(splitLen));
    const uint32_t copyPitch = AlignUp(copyLen * sizeof(T), BLOCK_SIZE) / sizeof(T);
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(outerReal), static_cast<uint32_t>(copyLen * sizeof(T)),
        static_cast<uint32_t>((copyPitch - AlignUp(copyLen * sizeof(T), BLOCK_SIZE) / sizeof(T)) * sizeof(T)),
        static_cast<uint32_t>((splitLen - copyLen) * sizeof(T)), 0};
    DataCopyPad(outputGm[outerBase * static_cast<uint64_t>(splitLen) + colOff], inputLocal, copyParams);
    WaitMte3ToV();
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::ProcessFullRow512(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inputLocal = CopyInFullRow(outerBase, outerReal);
    FirstVnchw512Parity(inputLocal, transTensor, rowLength);
    for (uint32_t g = 0; g < splitNum; ++g) {
        const uint32_t splitLen = sizeSplits[g];
        CompactSegment512(inputLocal, splitStarts[g], splitLen, rowLength);
        LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        SecondVnchw512Parity(inputLocal, outputLocal, splitLen);
        outQueue.EnQue(outputLocal);
        LocalTensor<T> resultLocal = outQueue.DeQue<T>();
        CopyOutContiguous(resultLocal, g, outerBase, outerReal, splitLen);
        outQueue.FreeTensor(resultLocal);
    }
    inQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::ProcessFullRow256Virtual(uint64_t outerBase, uint32_t outerReal)
{
    LocalTensor<T> inputLocal = CopyInFullRow(outerBase, outerReal);
    FirstVnchw256(inputLocal, transTensor);
    for (uint32_t v = 0; v < virtualSplitNum; ++v) {
        const uint32_t vStart = v * virtualSplitSize;
        const uint32_t vEnd = vStart + virtualSplitSize;
        CompactSegment256(inputLocal, vStart, virtualSplitSize);
        LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        SecondVnchw256(inputLocal, outputLocal, virtualSplitSize);
        outQueue.EnQue(outputLocal);
        LocalTensor<T> resultLocal = outQueue.DeQue<T>();
        for (uint32_t g = 0; g < splitNum; ++g) {
            const uint32_t sStart = splitStarts[g];
            const uint32_t sEnd = sStart + sizeSplits[g];
            const uint32_t overlapStart = vStart > sStart ? vStart : sStart;
            const uint32_t overlapEnd = vEnd < sEnd ? vEnd : sEnd;
            if (overlapStart < overlapEnd) {
                CopyOutOverlap(resultLocal, g, outerBase, outerReal, sizeSplits[g], virtualSplitSize,
                               overlapStart - vStart, overlapStart - sStart, overlapEnd - overlapStart);
            }
        }
        outQueue.FreeTensor(resultLocal);
    }
    inQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::ProcessChunk512(uint64_t taskIdx)
{
    const uint64_t outerTileIdx = taskIdx / colChunkNum;
    const uint32_t chunkIdx = static_cast<uint32_t>(taskIdx % colChunkNum);
    const uint64_t outerBase = outerTileIdx * static_cast<uint64_t>(outerTile);
    const uint32_t outerReal = (outerTileIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
    if (chunkIdx >= splitNum) {
        return;
    }
    const uint32_t colBase = splitStarts[chunkIdx];
    const uint32_t chunkCols = sizeSplits[chunkIdx];
    const uint32_t chunkRowPitch = rowTransLen / GROUP_ROWS_512;

    LocalTensor<T> inputLocal = CopyInChunk(outerBase, outerReal, colBase, chunkCols);
    FirstVnchw512Parity(inputLocal, transTensor, chunkRowPitch);
    CompactSegment512(inputLocal, 0, chunkCols, chunkRowPitch);
    LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
    SecondVnchw512Parity(inputLocal, outputLocal, chunkCols);
    outQueue.EnQue(outputLocal);
    LocalTensor<T> resultLocal = outQueue.DeQue<T>();
    CopyOutContiguous(resultLocal, chunkIdx, outerBase, outerReal, chunkCols);
    outQueue.FreeTensor(resultLocal);
    inQueue.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::ProcessPureTile(uint64_t outerBase, uint32_t outerReal)
{
    for (uint32_t g = 0; g < splitNum; ++g) {
        const uint32_t splitLen = sizeSplits[g];
        for (uint32_t colOff = 0; colOff < splitLen;) {
            const uint32_t remain = splitLen - colOff;
            const uint32_t copyLen = remain > PURE_COPY_COL_LIMIT ? PURE_COPY_COL_LIMIT : remain;
            LocalTensor<T> inputLocal = CopyInPure(outerBase, outerReal, g, colOff, copyLen);
            CopyOutPure(inputLocal, g, outerBase, outerReal, splitLen, colOff, copyLen);
            inQueue.FreeTensor(inputLocal);
            colOff += copyLen;
        }
    }
}

template <typename T>
__aicore__ inline void SplitVUnevenCompact8Bit<T>::Process()
{
    if (!IsValidTiling()) {
        return;
    }

    for (uint64_t i = 0; i < tileNum; ++i) {
        const uint64_t taskIdx = loopOff + i;
        if (mode == MODE_CHUNK_512_PARITY) {
            ProcessChunk512(taskIdx);
            continue;
        }
        const uint64_t outerBase = taskIdx * static_cast<uint64_t>(outerTile);
        const uint32_t outerReal = (taskIdx == static_cast<uint64_t>(outerTileNum - 1)) ? outerTail : outerTile;
        if (mode == MODE_FULL_512_PARITY) {
            ProcessFullRow512(outerBase, outerReal);
        } else if (mode == MODE_FULL_256_VIRTUAL) {
            ProcessFullRow256Virtual(outerBase, outerReal);
        } else {
            ProcessPureTile(outerBase, outerReal);
        }
    }
}

#endif // OP_KERNEL_SPLIT_V_UNEVEN_COMPACT_8BIT_H_
