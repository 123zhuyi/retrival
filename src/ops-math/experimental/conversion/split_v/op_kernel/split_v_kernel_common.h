/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_KERNEL_SPLIT_V_KERNEL_COMMON_H_
#define OP_KERNEL_SPLIT_V_KERNEL_COMMON_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "split_v_tiling_data.h"

using namespace AscendC;

template <typename ValueT, typename AlignT>
__aicore__ inline ValueT SplitVAlignUp(ValueT value, AlignT align)
{
    const ValueT typedAlign = static_cast<ValueT>(align);
    if (typedAlign == static_cast<ValueT>(0)) {
        return value;
    }
    return ((value + typedAlign - static_cast<ValueT>(1)) / typedAlign) * typedAlign;
}

template <typename BytesT, typename AlignT>
__aicore__ inline bool SplitVIsAlignedBytes(BytesT bytes, AlignT align)
{
    const uint64_t alignValue = static_cast<uint64_t>(align);
    return alignValue != 0 && static_cast<uint64_t>(bytes) % alignValue == 0;
}

template <HardEvent event>
__aicore__ inline void SplitVSync()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

struct SplitVCoreRange {
    uint64_t base = 0;
    uint32_t rows = 0;
};

struct SplitVTaskRange {
    uint64_t base = 0;
    uint64_t count = 0;
};

__aicore__ inline uint32_t SplitVCalcParityOffset(uint32_t index, uint32_t pitch, bool oddGroup, bool highHalf,
                                                  bool halfShifted, uint32_t groupRows);

template <typename VT, uint32_t TRANS_BLOCK>
__aicore__ inline void SplitVSecondVnchw(LocalTensor<VT> src, LocalTensor<VT> dst, uint32_t curSplitSize);

#define SPLIT_V_DECLARE_TILE_COPY_API(CLASS_NAME)                                                      \
public:                                                                                                \
    __aicore__ inline CLASS_NAME() {}                                                                  \
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SplitVTilingData* tiling, TPipe* pipe); \
    __aicore__ inline void Process();                                                                  \
                                                                                                       \
private:                                                                                               \
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length, uint32_t alignedLength);           \
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length, uint32_t alignedLength)

struct SplitVChunkCopyInfo {
    bool valid = false;
    uint64_t copyElems = 0;
    uint32_t copyBytes = 0;
    uint32_t localPitchBytes = 0;
    uint32_t rightPadding = 0;
};

__aicore__ inline SplitVChunkCopyInfo SplitVMakeChunkCopyInfo(uint64_t splitElems, uint64_t chunkOffset,
                                                              uint32_t chunkElems, uint32_t elemSize,
                                                              uint32_t alignBytes)
{
    SplitVChunkCopyInfo info;
    if (chunkOffset >= splitElems || elemSize == 0) {
        return info;
    }
    info.valid = true;
    info.copyElems = splitElems - chunkOffset;
    if (info.copyElems > chunkElems) {
        info.copyElems = chunkElems;
    }
    info.copyBytes = static_cast<uint32_t>(info.copyElems * elemSize);
    info.localPitchBytes = static_cast<uint32_t>(SplitVAlignUp(info.copyBytes, alignBytes));
    info.rightPadding = static_cast<uint32_t>((info.localPitchBytes - info.copyBytes) / elemSize);
    return info;
}

template <typename T>
__aicore__ inline void SplitVFirstVnchw256Parity(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                                 uint32_t rowTransLen, uint32_t rowLength, bool oddGroup,
                                                 bool pHighHalf, uint32_t addrCount, uint32_t elemsPerDb)
{
    TransDataTo5HDParams params;
    params.dstHighHalf = oddGroup;
    params.srcHighHalf = pHighHalf;
    params.repeatTimes = static_cast<uint8_t>(rowLength / 4);
    params.dstRepStride = params.repeatTimes == 1 ? 0 : elemsPerDb;
    params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;

    uint64_t dstList[16];
    uint64_t srcList[16];
    const uint32_t groupParity = oddGroup ? 1 : 0;
    const uint32_t pBase = pHighHalf ? addrCount : 0;
    for (uint32_t j = 0; j < addrCount; ++j) {
        srcList[j] = reinterpret_cast<uint64_t>(src[(2 * j + groupParity) * rowTransLen].GetPhyAddr());
        dstList[j] = reinterpret_cast<uint64_t>(dst[(pBase + j) * elemsPerDb].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstList, srcList, params);
}

template <typename T>
__aicore__ inline void SplitVFirstVnchw512Parity(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                                 uint32_t rowTransLen, uint32_t groupTransLen, uint32_t groupRows,
                                                 bool oddGroup, bool pHighHalf, bool halfShifted, uint32_t addrCount,
                                                 uint32_t elemsPerDb)
{
    const uint8_t repeatTimes = static_cast<uint8_t>(pHighHalf ? groupTransLen / 2 : (groupTransLen + 1) / 2);
    if (repeatTimes == 0) {
        return;
    }

    TransDataTo5HDParams params;
    params.dstHighHalf = oddGroup;
    params.srcHighHalf = halfShifted && oddGroup ? !pHighHalf : pHighHalf;
    params.repeatTimes = repeatTimes;
    params.dstRepStride = params.repeatTimes == 1 ? 0 : elemsPerDb;
    params.srcRepStride = params.repeatTimes == 1 ? 0 : 1;

    uint64_t dstList[16];
    uint64_t srcList[16];
    const uint32_t pBase = pHighHalf ? addrCount : 0;
    for (uint32_t j = 0; j < addrCount; ++j) {
        const uint32_t srcOffset = SplitVCalcParityOffset(j, rowTransLen, oddGroup, pHighHalf, halfShifted, groupRows);
        srcList[j] = reinterpret_cast<uint64_t>(src[srcOffset].GetPhyAddr());
        dstList[j] = reinterpret_cast<uint64_t>(dst[(pBase + j) * elemsPerDb].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstList, srcList, params);
}

template <typename T>
__aicore__ inline void SplitVSecondParityVnchw(const LocalTensor<T>& src, const LocalTensor<T>& dst,
                                               uint32_t curSplitTransLen, bool oddGroup, bool pHighHalf,
                                               bool halfShifted, uint8_t repeatTimes, uint32_t addrCount,
                                               uint32_t elemsPerDb)
{
    if (repeatTimes == 0) {
        return;
    }
    TransDataTo5HDParams params;
    params.srcHighHalf = oddGroup;
    params.dstHighHalf = halfShifted && oddGroup ? !pHighHalf : pHighHalf;
    params.repeatTimes = repeatTimes;
    params.dstRepStride = repeatTimes == 1 ? 0 : 1;
    params.srcRepStride = repeatTimes == 1 ? 0 : elemsPerDb;

    uint64_t dstList[16];
    uint64_t srcList[16];
    const uint32_t pBase = pHighHalf ? addrCount : 0;
    for (uint32_t j = 0; j < addrCount; ++j) {
        srcList[j] = reinterpret_cast<uint64_t>(src[(pBase + j) * elemsPerDb].GetPhyAddr());
        const uint32_t dstOffset = SplitVCalcParityOffset(j, curSplitTransLen, oddGroup, pHighHalf, halfShifted,
                                                          addrCount);
        dstList[j] = reinterpret_cast<uint64_t>(dst[dstOffset].GetPhyAddr());
    }
    TransDataTo5HD<T>(dstList, srcList, params);
}

#define SPLIT_V_COMPACT_CORE_FIELDS \
    uint64_t outerLength = 0;       \
    uint32_t rowLength = 0;         \
    uint32_t splitSize = 0;         \
    uint32_t tailSplitSize = 0;     \
    uint32_t splitNum = 0;          \
    uint32_t outerTile = 0;         \
    uint32_t outerTail = 0;         \
    uint32_t outerTileNum = 0;      \
    uint64_t tileNum = 0;           \
    uint64_t loopOff = 0;           \
    uint64_t coreOuterBase = 0;     \
    uint64_t coreOuterLength = 0;   \
    uint32_t alignedNum = 0

template <typename T, uint32_t BufferNum>
class SplitVCompactState {
protected:
    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, BufferNum> queue;
    TBuf<TPosition::VECCALC> transBuf;
    TBuf<TPosition::VECCALC> segBuf;
    LocalTensor<T> transTensor;
    LocalTensor<T> segTensor;

    uint64_t totalLength = 0;
    SPLIT_V_COMPACT_CORE_FIELDS;
    uint32_t rowPitch = 0;
    uint32_t chunkSplitNum = 0;
    uint32_t colChunkNum = 0;

    DataCopyExtParams copyInParam{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> copyInPadParam{false, 0, 0, 0};
    DataCopyParams segCopyParam{0, 0, 0, 0};

    __aicore__ inline void LoadSameLenCompactTiling(GM_ADDR output, SplitVTilingDataSameLenCompact* tiling,
                                                    uint32_t blockIdx, uint32_t transBlock)
    {
        outList.Init((__gm__ void*)output);
        totalLength = tiling->totalLength;
        outerLength = tiling->outerLength;
        rowLength = tiling->rowLength;
        splitSize = tiling->splitSize;
        tailSplitSize = tiling->tailSplitSize == 0 ? splitSize : tiling->tailSplitSize;
        splitNum = tiling->splitNum;
        outerTile = tiling->outerTile;
        outerTail = tiling->outerTail;
        outerTileNum = tiling->outerTileNum;
        chunkSplitNum = tiling->chunkSplitNum == 0 ? splitNum : tiling->chunkSplitNum;
        colChunkNum = tiling->colChunkNum == 0 ? 1 : tiling->colChunkNum;
        rowPitch = (tiling->rowTransLen == 0 || transBlock == 0) ? rowLength : tiling->rowTransLen / transBlock;
        alignedNum = 32 / sizeof(T);

        if (blockIdx < tiling->formerNum) {
            tileNum = tiling->formerOuterTileNum;
            loopOff = static_cast<uint64_t>(blockIdx) * tiling->formerOuterTileNum;
        } else {
            tileNum = tiling->tailOuterTileNum;
            loopOff = static_cast<uint64_t>(tiling->formerNum) * tiling->formerOuterTileNum +
                      static_cast<uint64_t>(blockIdx - tiling->formerNum) * tiling->tailOuterTileNum;
        }

        const uint64_t firstOuterTile = loopOff / colChunkNum;
        const uint64_t lastOuterTile = (loopOff + tileNum - 1) / colChunkNum;
        coreOuterBase = firstOuterTile * outerTile;
        const uint64_t coreOuterEnd = lastOuterTile == static_cast<uint64_t>(outerTileNum - 1) ?
                                          outerLength :
                                          (lastOuterTile + 1) * outerTile;
        coreOuterLength = coreOuterEnd - coreOuterBase;
    }

    __aicore__ inline void InitSameLenCompactBuffers(TPipe* pipe, uint32_t bufferNum, uint32_t inputElems,
                                                     uint32_t transElems, uint32_t splitElems)
    {
        pipe->InitBuffer(queue, bufferNum, inputElems * sizeof(T));
        pipe->InitBuffer(transBuf, transElems * sizeof(T));
        pipe->InitBuffer(segBuf, splitElems * sizeof(T));
        transTensor = transBuf.template Get<T>();
        segTensor = segBuf.template Get<T>();

        segCopyParam.blockCount = 16;
        segCopyParam.blockLen = splitSize;
        segCopyParam.srcStride = rowPitch - splitSize;
        segCopyParam.dstStride = 0;
    }

    __aicore__ inline LocalTensor<T> CopyInSameLenCompactTile(uint64_t outerBase, uint32_t outerReal)
    {
        LocalTensor<T> inLocalRaw = queue.template AllocTensor<T>();
        const uint32_t inputElems = outerReal * rowLength;
        const uint32_t inputCopyBytes = inputElems * sizeof(T);
        if (outerReal == outerTile && inputCopyBytes <= 65535U) {
            DataCopy(inLocalRaw, inputGm[outerBase * static_cast<uint64_t>(rowLength)], inputElems);
        } else {
            copyInParam.blockCount = 1;
            copyInParam.blockLen = inputCopyBytes;
            copyInParam.srcStride = 0;
            copyInParam.dstStride = 0;
            copyInPadParam.isPad = false;
            copyInPadParam.leftPadding = 0;
            copyInPadParam.rightPadding = 0;
            DataCopyPad(inLocalRaw, inputGm[outerBase * static_cast<uint64_t>(rowLength)], copyInParam, copyInPadParam);
        }
        queue.template EnQue<T>(inLocalRaw);
        return queue.template DeQue<T>();
    }

    __aicore__ inline void CopyOutSameLenCompactSegment(const LocalTensor<T>& inReadyRaw, uint64_t outputOffset,
                                                        uint32_t outputElems, uint32_t outputCopyElems)
    {
        const uint32_t outputCopyBytes = outputElems * sizeof(T);
        if (outputCopyBytes <= 65535U && outputCopyElems == outputElems) {
            DataCopy(outputGm[outputOffset], inReadyRaw, outputCopyElems);
        } else {
            copyInParam.blockCount = 1;
            copyInParam.blockLen = outputCopyBytes;
            copyInParam.srcStride = 0;
            copyInParam.dstStride = 0;
            DataCopyPad(outputGm[outputOffset], inReadyRaw, copyInParam);
        }
    }

    template <typename VT, uint32_t TRANS_BLOCK>
    __aicore__ inline uint32_t TransposeSameLenCompactSegment(LocalTensor<VT> segBits, LocalTensor<VT> transBits,
                                                              LocalTensor<VT> outBits, uint32_t splitIdx,
                                                              uint32_t curSplitSize, uint32_t outputElems)
    {
        const uint32_t outputCopyElems = SplitVAlignUp(outputElems, alignedNum);
        const uint32_t srcOffset = splitIdx * splitSize * TRANS_BLOCK;
        segCopyParam.blockLen = curSplitSize;
        segCopyParam.srcStride = rowPitch - curSplitSize;
        SplitVSync<HardEvent::V_MTE2>();
        DataCopy(segBits, transBits[srcOffset], segCopyParam);
        SplitVSync<HardEvent::MTE2_V>();

        SplitVSecondVnchw<VT, TRANS_BLOCK>(segBits, outBits, curSplitSize);
        SplitVSync<HardEvent::V_MTE3>();
        return outputCopyElems;
    }
};

#define SPLIT_V_USE_COMPACT_STATE(BASE_TYPE)       \
    using BASE_TYPE::outList;                      \
    using BASE_TYPE::inputGm;                      \
    using BASE_TYPE::outputGm;                     \
    using BASE_TYPE::queue;                        \
    using BASE_TYPE::transBuf;                     \
    using BASE_TYPE::segBuf;                       \
    using BASE_TYPE::transTensor;                  \
    using BASE_TYPE::segTensor;                    \
    using BASE_TYPE::totalLength;                  \
    using BASE_TYPE::outerLength;                  \
    using BASE_TYPE::rowLength;                    \
    using BASE_TYPE::splitSize;                    \
    using BASE_TYPE::tailSplitSize;                \
    using BASE_TYPE::splitNum;                     \
    using BASE_TYPE::outerTile;                    \
    using BASE_TYPE::outerTail;                    \
    using BASE_TYPE::outerTileNum;                 \
    using BASE_TYPE::tileNum;                      \
    using BASE_TYPE::loopOff;                      \
    using BASE_TYPE::coreOuterBase;                \
    using BASE_TYPE::coreOuterLength;              \
    using BASE_TYPE::alignedNum;                   \
    using BASE_TYPE::rowPitch;                     \
    using BASE_TYPE::chunkSplitNum;                \
    using BASE_TYPE::colChunkNum;                  \
    using BASE_TYPE::copyInParam;                  \
    using BASE_TYPE::copyInPadParam;               \
    using BASE_TYPE::segCopyParam;                 \
    using BASE_TYPE::LoadSameLenCompactTiling;     \
    using BASE_TYPE::InitSameLenCompactBuffers;    \
    using BASE_TYPE::CopyInSameLenCompactTile;     \
    using BASE_TYPE::CopyOutSameLenCompactSegment; \
    using BASE_TYPE::TransposeSameLenCompactSegment

class SplitVUnevenCompactTilingState {
protected:
    uint64_t totalLength = 0;
    uint64_t outerLength = 0;
    uint32_t rowLength = 0;
    uint32_t splitNum = 0;
    const uint32_t* sizeSplits = nullptr;
    const uint32_t* splitStarts = nullptr;
    uint32_t maxSplitSize = 0;
    uint32_t outerTile = 0;
    uint32_t outerTail = 0;
    uint32_t outerTileNum = 0;
    uint64_t tileNum = 0;
    uint64_t loopOff = 0;
    uint32_t mode = 0;
    uint32_t colChunkSize = 0;
    uint32_t colChunkNum = 0;

    __aicore__ inline void LoadUnevenCompactTiling(SplitVTilingDataUnevenCompact* tiling, uint32_t blockIdx)
    {
        totalLength = tiling->totalLength;
        outerLength = tiling->outerLength;
        rowLength = tiling->rowLength;
        splitNum = tiling->splitNum;
        sizeSplits = tiling->sizeSplits;
        splitStarts = tiling->splitStarts;
        maxSplitSize = tiling->maxSplitSize;
        outerTile = tiling->outerTile;
        outerTail = tiling->outerTail;
        outerTileNum = tiling->outerTileNum;
        mode = tiling->mode;
        colChunkSize = tiling->colChunkSize;
        colChunkNum = tiling->colChunkNum;

        if (blockIdx < tiling->formerNum) {
            tileNum = tiling->formerOuterTileNum;
            loopOff = static_cast<uint64_t>(blockIdx) * tiling->formerOuterTileNum;
        } else {
            tileNum = tiling->tailOuterTileNum;
            loopOff = static_cast<uint64_t>(tiling->formerNum) * tiling->formerOuterTileNum +
                      static_cast<uint64_t>(blockIdx - tiling->formerNum) * tiling->tailOuterTileNum;
        }
    }
};

#define SPLIT_V_USE_UNEVEN_COMPACT_TILING_STATE(BASE_TYPE) \
    using BASE_TYPE::totalLength;                          \
    using BASE_TYPE::outerLength;                          \
    using BASE_TYPE::rowLength;                            \
    using BASE_TYPE::splitNum;                             \
    using BASE_TYPE::sizeSplits;                           \
    using BASE_TYPE::splitStarts;                          \
    using BASE_TYPE::maxSplitSize;                         \
    using BASE_TYPE::outerTile;                            \
    using BASE_TYPE::outerTail;                            \
    using BASE_TYPE::outerTileNum;                         \
    using BASE_TYPE::tileNum;                              \
    using BASE_TYPE::loopOff;                              \
    using BASE_TYPE::mode;                                 \
    using BASE_TYPE::colChunkSize;                         \
    using BASE_TYPE::colChunkNum;                          \
    using BASE_TYPE::LoadUnevenCompactTiling

template <typename T, uint32_t QueueDepth>
class SplitVPureCopyState {
protected:
    uint32_t blockIdx = 0;
    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QueueDepth> queue;

    uint64_t totalLength = 0;
    uint64_t outerLength = 0;
    uint32_t rowLength = 0;
    uint32_t mode = 0;
    uint32_t splitPitch = 0;
    uint32_t realCoreNum = 0;
    uint32_t formerCoreRows = 0;
    uint32_t tailCoreRows = 0;
    uint32_t formerNum = 0;
    uint32_t outerTile = 0;
    uint32_t colTileLength = 0;
    uint32_t colTilePitch = 0;

    uint64_t coreOuterBase = 0;
    uint32_t coreRows = 0;

    template <typename TilingDataT>
    __aicore__ inline void LoadPureCopySchedule(const TilingDataT* tilingData)
    {
        mode = tilingData->mode;
        splitPitch = tilingData->splitPitch;
        realCoreNum = tilingData->realCoreNum;
        formerCoreRows = tilingData->formerCoreRows;
        tailCoreRows = tilingData->tailCoreRows;
        formerNum = tilingData->formerNum;
        outerTile = tilingData->outerTile;
        colTileLength = tilingData->colTileLength;
        colTilePitch = tilingData->colTilePitch;
    }

    __aicore__ inline void InitPureCopyBuffer(GM_ADDR x, GM_ADDR yList, uint32_t bufferElems, TPipe* pipe)
    {
        inputGm.SetGlobalBuffer((__gm__ T*)x, totalLength);
        outList.Init((__gm__ void*)yList);
        pipe->InitBuffer(queue, 2, bufferElems * sizeof(T));
    }

    __aicore__ inline bool PreparePureCopyCore(bool validTiling)
    {
        if (blockIdx >= realCoreNum || !validTiling) {
            return false;
        }
        CalcCoreRows();
        return coreRows != 0;
    }

    __aicore__ inline void CalcCoreRows()
    {
        if (blockIdx < formerNum) {
            coreRows = formerCoreRows;
            coreOuterBase = static_cast<uint64_t>(blockIdx) * formerCoreRows;
        } else {
            coreRows = tailCoreRows;
            coreOuterBase = static_cast<uint64_t>(formerNum) * formerCoreRows +
                            static_cast<uint64_t>(blockIdx - formerNum) * tailCoreRows;
        }
    }

    __aicore__ inline LocalTensor<T> CopyInPureRowChunk(uint64_t srcOffset, uint32_t copyElems, uint32_t localPitch)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        const uint32_t rightPadding = localPitch - copyElems;
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(copyElems * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams = {rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), 0};
        DataCopyPad(local, inputGm[srcOffset], copyParams, padParams);
        queue.template EnQue<T>(local);
        return queue.template DeQue<T>();
    }

    __aicore__ inline void CopyOutPureRowChunk(const LocalTensor<T>& local, GlobalTensor<T>& targetGm,
                                               uint64_t dstOffset, uint32_t copyElems)
    {
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(copyElems * sizeof(T)), 0, 0, 0};
        DataCopyPad(targetGm[dstOffset], local, copyParams);
    }

    __aicore__ inline bool IsPureCopyScheduleValid(uint32_t maxCopySize, bool splitMajorPitchValid,
                                                   bool rowChunkPitchValid) const
    {
        if (outerLength == 0 || rowLength == 0 || maxCopySize == 0 || realCoreNum == 0 || realCoreNum > GetBlockNum()) {
            return false;
        }
        if (mode == 0) {
            return outerTile != 0 && splitMajorPitchValid;
        }
        return mode == 1 && colTileLength != 0 && colTilePitch >= colTileLength && rowChunkPitchValid;
    }

    __aicore__ inline LocalTensor<T> CopyInPureSplitMajor(uint64_t outerBase, uint32_t outerReal, uint32_t splitBase,
                                                          uint32_t splitLen, uint32_t localPitch)
    {
        LocalTensor<T> local = queue.template AllocTensor<T>();
        const uint32_t splitBytes = splitLen * sizeof(T);
        const uint32_t rightPadding = localPitch - splitLen;
        DataCopyExtParams copyParams = {static_cast<uint16_t>(outerReal), splitBytes,
                                        static_cast<uint32_t>((rowLength - splitLen) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParams = {rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), 0};
        const uint64_t srcOffset = outerBase * static_cast<uint64_t>(rowLength) + splitBase;
        DataCopyPad(local, inputGm[srcOffset], copyParams, padParams);
        queue.template EnQue<T>(local);
        return queue.template DeQue<T>();
    }

    __aicore__ inline void CopyOutPureSplitMajor(const LocalTensor<T>& local, uint64_t outerBase, uint32_t outerReal,
                                                 uint32_t splitIndex, uint32_t splitLen)
    {
        outputGm.SetGlobalBuffer(outList.template GetDataPtr<__gm__ T>(splitIndex),
                                 outerLength * static_cast<uint64_t>(splitLen));
        const uint64_t dstOffset = outerBase * static_cast<uint64_t>(splitLen);
        DataCopyExtParams copyParams = {static_cast<uint16_t>(outerReal), static_cast<uint32_t>(splitLen * sizeof(T)),
                                        0, 0, 0};
        DataCopyPad(outputGm[dstOffset], local, copyParams);
    }
};

#define SPLIT_V_USE_PURE_COPY_STATE(BASE_TYPE) \
    using BASE_TYPE::blockIdx;                 \
    using BASE_TYPE::outList;                  \
    using BASE_TYPE::inputGm;                  \
    using BASE_TYPE::outputGm;                 \
    using BASE_TYPE::queue;                    \
    using BASE_TYPE::totalLength;              \
    using BASE_TYPE::outerLength;              \
    using BASE_TYPE::rowLength;                \
    using BASE_TYPE::mode;                     \
    using BASE_TYPE::splitPitch;               \
    using BASE_TYPE::realCoreNum;              \
    using BASE_TYPE::formerCoreRows;           \
    using BASE_TYPE::tailCoreRows;             \
    using BASE_TYPE::formerNum;                \
    using BASE_TYPE::outerTile;                \
    using BASE_TYPE::colTileLength;            \
    using BASE_TYPE::colTilePitch;             \
    using BASE_TYPE::coreOuterBase;            \
    using BASE_TYPE::coreRows;                 \
    using BASE_TYPE::LoadPureCopySchedule;     \
    using BASE_TYPE::InitPureCopyBuffer;       \
    using BASE_TYPE::PreparePureCopyCore;      \
    using BASE_TYPE::CopyInPureRowChunk;       \
    using BASE_TYPE::CopyOutPureRowChunk;      \
    using BASE_TYPE::IsPureCopyScheduleValid;  \
    using BASE_TYPE::CopyInPureSplitMajor;     \
    using BASE_TYPE::CopyOutPureSplitMajor

template <typename T, uint32_t QueueDepth>
class SplitVInnerCopyState {
protected:
    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, QueueDepth> queue;

    uint64_t outerLength = 0;
    uint64_t midLength = 0;
    uint64_t innerLength = 0;
    uint32_t mode = 0;
    uint32_t outerTile = 0;
    uint32_t outerTileNum = 0;
    uint32_t outerTail = 0;
    uint32_t midTile = 0;
    uint32_t midTileNum = 0;
    uint32_t midTail = 0;
    uint32_t chunkElems = 0;
    uint32_t chunkElemsAligned = 0;
    uint32_t chunkNumMax = 0;
    uint64_t taskNum = 0;
    uint64_t taskOffset = 0;

    DataCopyExtParams copyInParam{0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParam{false, 0, 0, 0};
    DataCopyExtParams copyOutParam{0, 0, 0, 0, 0};

    __aicore__ inline LocalTensor<T> CopyInInnerChunk(uint64_t srcOffset, uint32_t outerReal,
                                                      const SplitVChunkCopyInfo& chunkInfo)
    {
        DataCopyPadExtParams<T> chunkPadParam{chunkInfo.rightPadding != 0, 0,
                                              static_cast<uint8_t>(chunkInfo.rightPadding), 0};

        LocalTensor<T> inLocal = queue.template AllocTensor<T>();
        copyInParam.blockCount = static_cast<uint16_t>(outerReal);
        copyInParam.blockLen = chunkInfo.copyBytes;
        copyInParam.srcStride = static_cast<uint32_t>((midLength * innerLength - chunkInfo.copyElems) * sizeof(T));
        copyInParam.dstStride = 0;
        DataCopyPad(inLocal, inputGm[srcOffset], copyInParam, chunkPadParam);
        queue.template EnQue<T>(inLocal);
        return queue.template DeQue<T>();
    }
};

#define SPLIT_V_USE_INNER_COPY_STATE(BASE_TYPE) \
    using BASE_TYPE::outList;                   \
    using BASE_TYPE::inputGm;                   \
    using BASE_TYPE::outputGm;                  \
    using BASE_TYPE::queue;                     \
    using BASE_TYPE::outerLength;               \
    using BASE_TYPE::midLength;                 \
    using BASE_TYPE::innerLength;               \
    using BASE_TYPE::mode;                      \
    using BASE_TYPE::outerTile;                 \
    using BASE_TYPE::outerTileNum;              \
    using BASE_TYPE::outerTail;                 \
    using BASE_TYPE::midTile;                   \
    using BASE_TYPE::midTileNum;                \
    using BASE_TYPE::midTail;                   \
    using BASE_TYPE::chunkElems;                \
    using BASE_TYPE::chunkElemsAligned;         \
    using BASE_TYPE::chunkNumMax;               \
    using BASE_TYPE::taskNum;                   \
    using BASE_TYPE::taskOffset;                \
    using BASE_TYPE::copyInParam;               \
    using BASE_TYPE::padParam;                  \
    using BASE_TYPE::copyOutParam;              \
    using BASE_TYPE::CopyInInnerChunk

template <typename T>
class SplitVTileCopyState {
protected:
    ListTensorDesc outList;
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> queue;

    uint64_t outerLength = 0;
    uint64_t midLength = 0;
    uint64_t innerLength = 0;
    uint32_t tileLength = 0;
    uint32_t tileNum = 0;
    uint32_t lastTileLength = 0;
    uint32_t alignedNum = 0;
    uint64_t loopNum = 0;
    uint64_t loopOffset = 0;
    int64_t splitNum = 0;
    const int64_t* sizeSplits = nullptr;
    uint32_t blockIdx = 0;

    __aicore__ inline void LoadTileCopyTiling(GM_ADDR output, SplitVTilingData* tiling)
    {
        outList.Init((__gm__ void*)output);
        splitNum = tiling->splitNum;
        sizeSplits = tiling->sizeSplits;
        tileLength = tiling->innerTileLength;
        tileNum = tiling->innerTileNum;
        lastTileLength = tiling->innerLastTileLength;
        outerLength = tiling->outerLength;
        midLength = tiling->midLength;
        innerLength = tiling->innerLength;
    }

    __aicore__ inline void LoadTileCopyLoop(SplitVTilingData* tiling)
    {
        if (blockIdx < tiling->formerNum) {
            loopNum = tiling->formerLoop;
            loopOffset = static_cast<uint64_t>(blockIdx) * tiling->formerLoop;
        } else {
            loopNum = tiling->tailLoop;
            loopOffset = static_cast<uint64_t>(tiling->formerNum) * tiling->formerLoop +
                         static_cast<uint64_t>(blockIdx - tiling->formerNum) * tiling->tailLoop;
        }
    }

    __aicore__ inline void InitTileCopyBuffer(GM_ADDR input, uint64_t inputElems, TPipe* pipe)
    {
        inputGm.SetGlobalBuffer((__gm__ T*)input, inputElems);
        alignedNum = 32 / sizeof(T);
        if (alignedNum == 0) {
            alignedNum = 1;
        }
        const uint32_t queueLength = SplitVAlignUp(tileLength, alignedNum);
        pipe->InitBuffer(queue, 2, queueLength * sizeof(T));
    }
};

#define SPLIT_V_USE_TILE_COPY_STATE(BASE_TYPE) \
    using BASE_TYPE::outList;                  \
    using BASE_TYPE::inputGm;                  \
    using BASE_TYPE::outputGm;                 \
    using BASE_TYPE::queue;                    \
    using BASE_TYPE::outerLength;              \
    using BASE_TYPE::midLength;                \
    using BASE_TYPE::innerLength;              \
    using BASE_TYPE::tileLength;               \
    using BASE_TYPE::tileNum;                  \
    using BASE_TYPE::lastTileLength;           \
    using BASE_TYPE::alignedNum;               \
    using BASE_TYPE::loopNum;                  \
    using BASE_TYPE::loopOffset;               \
    using BASE_TYPE::splitNum;                 \
    using BASE_TYPE::sizeSplits;               \
    using BASE_TYPE::blockIdx;                 \
    using BASE_TYPE::LoadTileCopyTiling;       \
    using BASE_TYPE::LoadTileCopyLoop;         \
    using BASE_TYPE::InitTileCopyBuffer

__aicore__ inline SplitVCoreRange SplitVCalcCoreRange(uint32_t blockIdx, uint32_t formerNum, uint32_t formerRows,
                                                      uint32_t tailRows)
{
    SplitVCoreRange range;
    if (blockIdx < formerNum) {
        range.rows = formerRows;
        range.base = static_cast<uint64_t>(blockIdx) * formerRows;
    } else {
        range.rows = tailRows;
        range.base = static_cast<uint64_t>(formerNum) * formerRows +
                     static_cast<uint64_t>(blockIdx - formerNum) * tailRows;
    }
    return range;
}

__aicore__ inline SplitVTaskRange SplitVCalcTaskRange(uint32_t blockIdx, uint64_t formerNum, uint64_t formerTaskNum,
                                                      uint64_t tailTaskNum)
{
    SplitVTaskRange range;
    if (static_cast<uint64_t>(blockIdx) < formerNum) {
        range.count = formerTaskNum;
        range.base = static_cast<uint64_t>(blockIdx) * formerTaskNum;
    } else {
        range.count = tailTaskNum;
        range.base = formerNum * formerTaskNum + (static_cast<uint64_t>(blockIdx) - formerNum) * tailTaskNum;
    }
    return range;
}

__aicore__ inline uint32_t SplitVSelectTileReal(uint64_t tileIdx, uint32_t tileNum, uint32_t tailSize,
                                                uint32_t tileSize)
{
    return tileIdx == static_cast<uint64_t>(tileNum - 1) ? tailSize : tileSize;
}

__aicore__ inline uint32_t SplitVSelectOuterReal(uint64_t tileIdx, uint32_t outerTileNum, uint32_t outerTail,
                                                 uint32_t outerTile)
{
    return SplitVSelectTileReal(tileIdx, outerTileNum, outerTail, outerTile);
}

template <typename VT, uint32_t TRANS_BLOCK>
__aicore__ inline void SplitVFirstVnchw(LocalTensor<VT> src, LocalTensor<VT> dst, uint32_t curRowPitch)
{
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = static_cast<uint8_t>(curRowPitch);
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : TRANS_BLOCK;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;

    uint64_t srcList[TRANS_BLOCK];
    uint64_t dstList[TRANS_BLOCK];
    for (uint32_t j = 0; j < TRANS_BLOCK; ++j) {
        srcList[j] = reinterpret_cast<uint64_t>(src[static_cast<uint32_t>(j) * TRANS_BLOCK * curRowPitch].GetPhyAddr());
        dstList[j] = reinterpret_cast<uint64_t>(dst[static_cast<uint32_t>(j) * TRANS_BLOCK].GetPhyAddr());
    }
    TransDataTo5HD<VT>(dstList, srcList, transDataParams);
    PipeBarrier<PIPE_V>();
}

template <typename VT, uint32_t TRANS_BLOCK>
__aicore__ inline void SplitVSecondVnchw(LocalTensor<VT> src, LocalTensor<VT> dst, uint32_t curSplitSize)
{
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = static_cast<uint8_t>(curSplitSize);
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : TRANS_BLOCK;

    uint64_t srcList[TRANS_BLOCK];
    uint64_t dstList[TRANS_BLOCK];
    for (uint32_t j = 0; j < TRANS_BLOCK; ++j) {
        srcList[j] = reinterpret_cast<uint64_t>(src[j * TRANS_BLOCK].GetPhyAddr());
        dstList[j] = reinterpret_cast<uint64_t>(dst[j * TRANS_BLOCK * curSplitSize].GetPhyAddr());
    }
    TransDataTo5HD<VT>(dstList, srcList, transDataParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline DataCopyExtParams SplitVMakeCopyParams(uint16_t blockCount, uint32_t blockLen, uint32_t srcStride,
                                                         uint32_t dstStride, uint32_t rsv = 0)
{
    DataCopyExtParams copyParams = {blockCount, blockLen, srcStride, dstStride, rsv};
    return copyParams;
}

template <typename T>
__aicore__ inline DataCopyPadExtParams<T> SplitVMakePadParams(bool isPad, uint8_t leftPadding, uint8_t rightPadding,
                                                              T paddingValue = 0)
{
    DataCopyPadExtParams<T> padParams = {isPad, leftPadding, rightPadding, paddingValue};
    return padParams;
}

template <typename T, typename QueueT>
__aicore__ inline void SplitVQueueCopyIn(QueueT& queue, GlobalTensor<T>& inputGm, uint64_t offset, uint32_t length,
                                         uint32_t alignedLength)
{
    LocalTensor<T> inLocal = queue.template AllocTensor<T>();
    if (length == alignedLength) {
        DataCopy(inLocal, inputGm[offset], alignedLength);
    } else {
        DataCopyParams dataCopyParams = {1, 0, 0, 0};
        DataCopyPadParams dataCopyPadParams = {true, 0, 0, 0};
        dataCopyParams.blockLen = length * sizeof(T);
        dataCopyPadParams.rightPadding = alignedLength - length;
        DataCopyPad(inLocal, inputGm[offset], dataCopyParams, dataCopyPadParams);
    }
    queue.template EnQue<T>(inLocal);
}

template <typename T, typename QueueT>
__aicore__ inline void SplitVQueueCopyOut(QueueT& queue, GlobalTensor<T>& outputGm, uint64_t offset, uint32_t length,
                                          uint32_t alignedLength)
{
    LocalTensor<T> outLocal = queue.template DeQue<T>();
    if (length == alignedLength) {
        DataCopy(outputGm[offset], outLocal, length);
    } else {
        DataCopyParams dataCopyParams = {1, 0, 0, 0};
        dataCopyParams.blockLen = length * sizeof(T);
        DataCopyPad(outputGm[offset], outLocal, dataCopyParams);
    }
    queue.FreeTensor(outLocal);
}

__aicore__ inline uint32_t SplitVCalcParityOffset(uint32_t index, uint32_t pitch, bool oddGroup, bool highHalf,
                                                  bool halfShifted, uint32_t halfShift)
{
    if (!halfShifted) {
        return (2 * index + (oddGroup ? 1U : 0U)) * pitch;
    }
    if (!oddGroup) {
        return 2 * index * pitch;
    }
    if (!highHalf) {
        return (2 * index + 1U) * pitch - halfShift;
    }
    return (2 * index + 1U) * pitch + halfShift;
}

#endif // OP_KERNEL_SPLIT_V_KERNEL_COMMON_H_
