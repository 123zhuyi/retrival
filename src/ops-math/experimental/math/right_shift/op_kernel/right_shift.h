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
 * \file right_shift.h
 * \brief RightShift kernel
 */

#ifndef RIGHT_SHIFT_H
#define RIGHT_SHIFT_H

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "right_shift_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "right_shift_tiling_key.h"
#endif

namespace NsRightShift {
using namespace AscendC;

constexpr uint32_t DATABLOCK_BYTES = 32;
constexpr uint32_t BROADCAST_SCALAR_BUFFER_BYTES = 32;

template <typename T>
struct IsVectorShiftType {
    static constexpr bool value = std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
                                  std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value;
};

template <typename T, uint32_t BROADCAST_MODE>
class RightShiftKernel {
public:
    __aicore__ inline RightShiftKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const RightShiftTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint64_t CalcOffset(uint64_t outIndex, const uint64_t* stride) const;
    __aicore__ inline void InitOutputCoord(uint64_t outIndex, uint64_t* coord) const;
    __aicore__ inline void AdvanceTailSegment(uint64_t* coord, uint64_t& xOffset, uint64_t& yOffset) const;
    __aicore__ inline bool IsInvalidShift(T shiftValue) const;
    __aicore__ inline uint32_t NormalizeShiftBits(T shiftValue) const;
    __aicore__ inline T InvalidShiftValue(T xValue) const;
    __aicore__ inline T ComputeOne(T xValue, T yValue) const;
    __aicore__ inline bool IsContiguousStride(const uint64_t* stride) const;
    __aicore__ inline bool IsOuterRowBroadcast(const uint64_t* stride) const;
    __aicore__ inline bool IsOuterBroadcast(const uint64_t* broadcastStride, const uint64_t* contiguousStride) const;
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const;
    __aicore__ inline uint32_t CalcBroadcastSegmentLength(uint64_t outStart, uint64_t remainLength) const;
    __aicore__ inline uint32_t CalcPackedBroadcastLength(uint64_t remainLength) const;
    __aicore__ inline bool CanUsePackedBroadcast(uint64_t outStart) const;
    __aicore__ inline void BuildCopyParams(uint32_t count, DataCopyParams& copyParams) const;
    __aicore__ inline void FillLocal(LocalTensor<T> dst, T value, uint32_t count);
    __aicore__ inline void CopyContiguousIn(LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t offset, uint32_t count);
    __aicore__ inline void CopyContiguousOut(GlobalTensor<T>& dst, uint64_t offset, LocalTensor<T> src, uint32_t count);
    __aicore__ inline void BroadcastScalarToLocal(LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset,
                                                  uint32_t length, uint32_t fillLength);
    __aicore__ inline void CopyBroadcastSegmentByOffset(LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset,
                                                        uint64_t tailStride, uint32_t length, uint32_t fillLength);
    __aicore__ inline void CopyBroadcastSegment(LocalTensor<T> dst, GlobalTensor<T>& src, const uint64_t* stride,
                                                uint64_t outStart, uint32_t length, uint32_t fillLength);
    __aicore__ inline void BroadcastRepeatedRowsToLocal(LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset,
                                                        uint32_t rowLength, uint32_t rowCount);
    __aicore__ inline void CopyBroadcastInputs(LocalTensor<T> xLocal, LocalTensor<T> yLocal, uint64_t outStart,
                                               uint32_t validLength, uint32_t computeLength);
    __aicore__ inline void CopyBroadcastInputsPacked(LocalTensor<T> xLocal, LocalTensor<T> yLocal, uint64_t outStart,
                                                     uint32_t validLength, uint32_t computeLength);
    __aicore__ inline T ReadScalarY();
    __aicore__ inline void InitInvalidShiftResult(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count);
    __aicore__ inline void CastShiftToCompare(LocalTensor<T> ySrc, LocalTensor<half> yCmp, uint32_t count);
    __aicore__ inline void CompareHalfShift(LocalTensor<uint8_t> maskLocal, LocalTensor<half> yCmp, uint32_t shift,
                                            uint32_t count);
    __aicore__ inline void ApplyVectorBucketShift(LocalTensor<T> dst, LocalTensor<T> xSrc, LocalTensor<T> ySrc,
                                                  uint32_t count);
    __aicore__ inline void ComputeScalarShift(LocalTensor<T> dst, LocalTensor<T> src, uint32_t count, T shiftValue);
    __aicore__ inline void ComputeElementwise(LocalTensor<T> dst, LocalTensor<T> xSrc, LocalTensor<T> ySrc,
                                              uint32_t count);
    __aicore__ inline void ProcessScalarY();
    __aicore__ inline void ProcessElementwise();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> yBuf;
    TBuf<QuePosition::VECCALC> zBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> cmpBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> singleBuf;
    TBuf<QuePosition::VECCALC> rowBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    uint64_t blockOffset = 0;
    uint64_t blockLength = 0;
    uint64_t totalLength = 0;
    uint64_t tileBufferLen = 0;
    uint32_t rank = 1;
    uint64_t outShape[RIGHT_SHIFT_MAX_BROADCAST_DIM] = {};
    uint64_t xStride[RIGHT_SHIFT_MAX_BROADCAST_DIM] = {};
    uint64_t yStride[RIGHT_SHIFT_MAX_BROADCAST_DIM] = {};
    bool xContiguous = false;
    bool yContiguous = false;
    bool useRowBroadcastBuffer = false;
};

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                                 const RightShiftTilingData* tilingData)
{
    uint64_t coreId = GetBlockIdx();
    uint64_t usedCoreNum = tilingData->formerCoreNum + tilingData->tailCoreNum;
    totalLength = tilingData->totalLength;
    tileBufferLen = tilingData->tileBufferLen;
    rank = tilingData->rank == 0 ? 1 : tilingData->rank;
    for (uint32_t i = 0; i < RIGHT_SHIFT_MAX_BROADCAST_DIM; ++i) {
        outShape[i] = tilingData->outShape[i];
        xStride[i] = tilingData->xStride[i];
        yStride[i] = tilingData->yStride[i];
    }

    if (coreId >= usedCoreNum) {
        blockLength = 0;
    } else if (coreId < tilingData->formerCoreNum) {
        blockLength = tilingData->formerCoreDataNum;
        blockOffset = coreId * tilingData->formerCoreDataNum;
    } else {
        blockLength = tilingData->tailCoreDataNum;
        blockOffset = tilingData->formerCoreNum * tilingData->formerCoreDataNum +
                      (coreId - tilingData->formerCoreNum) * tilingData->tailCoreDataNum;
    }

    if (blockOffset >= totalLength) {
        blockLength = 0;
    } else if (blockOffset + blockLength > totalLength) {
        blockLength = totalLength - blockOffset;
    }

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    zGm.SetGlobalBuffer((__gm__ T*)z);

    xContiguous = IsContiguousStride(xStride);
    yContiguous = IsContiguousStride(yStride);
    if (blockLength == 0 || tileBufferLen == 0) {
        return;
    }

    uint64_t rowBufferBytes = BROADCAST_SCALAR_BUFFER_BYTES;
    if (rank > 1) {
        uint64_t tailDim = outShape[rank - 1];
        bool rowBytesAligned = tailDim > 0 && (tailDim * sizeof(T) % DATABLOCK_BYTES) == 0;
        useRowBroadcastBuffer = tailDim > 0 && tailDim <= tileBufferLen && rowBytesAligned &&
                                (IsOuterBroadcast(xStride, yStride) || IsOuterBroadcast(yStride, xStride));
        if (useRowBroadcastBuffer) {
            rowBufferBytes = tailDim * sizeof(T);
        }
    }

    pipe.InitBuffer(xBuf, tileBufferLen * sizeof(T));
    pipe.InitBuffer(yBuf, tileBufferLen * sizeof(T));
    pipe.InitBuffer(zBuf, tileBufferLen * sizeof(T));
    pipe.InitBuffer(workBuf, tileBufferLen * sizeof(T));
    pipe.InitBuffer(cmpBuf, tileBufferLen * sizeof(int32_t));
    pipe.InitBuffer(maskBuf, 128);
    pipe.InitBuffer(singleBuf, BROADCAST_SCALAR_BUFFER_BYTES);
    pipe.InitBuffer(rowBuf, rowBufferBytes);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint64_t RightShiftKernel<T, BROADCAST_MODE>::CalcOffset(uint64_t outIndex,
                                                                           const uint64_t* stride) const
{
    uint64_t offset = 0;
    uint64_t remain = outIndex;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        uint32_t idx = static_cast<uint32_t>(i);
        uint64_t dim = outShape[idx];
        uint64_t coord = dim == 0 ? 0 : remain % dim;
        remain = dim == 0 ? 0 : remain / dim;
        offset += coord * stride[idx];
    }
    return offset;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::InitOutputCoord(uint64_t outIndex, uint64_t* coord) const
{
    uint64_t remain = outIndex;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        uint32_t idx = static_cast<uint32_t>(i);
        uint64_t dim = outShape[idx];
        coord[idx] = dim == 0 ? 0 : remain % dim;
        remain = dim == 0 ? 0 : remain / dim;
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::AdvanceTailSegment(uint64_t* coord, uint64_t& xOffset,
                                                                               uint64_t& yOffset) const
{
    if (rank <= 1) {
        return;
    }

    for (int32_t i = static_cast<int32_t>(rank) - 2; i >= 0; --i) {
        uint32_t idx = static_cast<uint32_t>(i);
        coord[idx]++;
        xOffset += xStride[idx];
        yOffset += yStride[idx];
        if (coord[idx] < outShape[idx]) {
            return;
        }

        coord[idx] = 0;
        xOffset -= outShape[idx] * xStride[idx];
        yOffset -= outShape[idx] * yStride[idx];
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool RightShiftKernel<T, BROADCAST_MODE>::IsInvalidShift(T shiftValue) const
{
    constexpr T maxValidShift = static_cast<T>(sizeof(T) * 8 - 1);
    if (shiftValue > maxValidShift) {
        return true;
    }
    if constexpr (std::is_signed<T>::value) {
        if (shiftValue < static_cast<T>(0)) {
            return true;
        }
    }
    return false;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint32_t RightShiftKernel<T, BROADCAST_MODE>::NormalizeShiftBits(T shiftValue) const
{
    return static_cast<uint32_t>(static_cast<uint64_t>(shiftValue));
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline T RightShiftKernel<T, BROADCAST_MODE>::InvalidShiftValue(T xValue) const
{
    if constexpr (std::is_signed<T>::value) {
        return xValue < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(0);
    }
    return static_cast<T>(0);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline T RightShiftKernel<T, BROADCAST_MODE>::ComputeOne(T xValue, T yValue) const
{
    if (IsInvalidShift(yValue)) {
        return InvalidShiftValue(xValue);
    }

    uint32_t shift = NormalizeShiftBits(yValue);
    if constexpr (std::is_signed<T>::value) {
        int64_t signedX = static_cast<int64_t>(xValue);
        return static_cast<T>(signedX >> shift);
    } else {
        uint64_t unsignedX = static_cast<uint64_t>(xValue);
        return static_cast<T>(unsignedX >> shift);
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool RightShiftKernel<T, BROADCAST_MODE>::IsContiguousStride(const uint64_t* stride) const
{
    uint64_t expectedStride = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        uint32_t idx = static_cast<uint32_t>(i);
        if (stride[idx] != expectedStride) {
            return false;
        }
        expectedStride *= outShape[idx];
    }
    return true;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool RightShiftKernel<T, BROADCAST_MODE>::IsOuterRowBroadcast(const uint64_t* stride) const
{
    if (rank == 0 || stride[rank - 1] != 1) {
        return false;
    }

    for (uint32_t i = 0; i + 1 < rank; ++i) {
        if (stride[i] != 0) {
            return false;
        }
    }
    return outShape[rank - 1] > 0;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool RightShiftKernel<T, BROADCAST_MODE>::IsOuterBroadcast(const uint64_t* broadcastStride,
                                                                             const uint64_t* contiguousStride) const
{
    return IsOuterRowBroadcast(broadcastStride) && IsContiguousStride(contiguousStride);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint32_t RightShiftKernel<T, BROADCAST_MODE>::AlignUp(uint32_t value, uint32_t align) const
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint32_t RightShiftKernel<T, BROADCAST_MODE>::CalcBroadcastSegmentLength(uint64_t outStart,
                                                                                           uint64_t remainLength) const
{
    uint64_t segmentLength = remainLength > tileBufferLen ? tileBufferLen : remainLength;
    if (rank > 0) {
        uint64_t tailDim = outShape[rank - 1];
        if (tailDim > 0) {
            uint64_t tailRemain = tailDim - outStart % tailDim;
            if (segmentLength > tailRemain) {
                segmentLength = tailRemain;
            }
        }
    }
    return static_cast<uint32_t>(segmentLength);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint32_t RightShiftKernel<T, BROADCAST_MODE>::CalcPackedBroadcastLength(uint64_t remainLength) const
{
    uint64_t tailDim = outShape[rank - 1];
    if (tailDim == 0) {
        return 0;
    }
    uint64_t maxLength = remainLength > tileBufferLen ? tileBufferLen : remainLength;
    return static_cast<uint32_t>(maxLength / tailDim * tailDim);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool RightShiftKernel<T, BROADCAST_MODE>::CanUsePackedBroadcast(uint64_t outStart) const
{
    if constexpr (BROADCAST_MODE != RIGHT_SHIFT_MODE_TAIL_CONTIGUOUS) {
        return false;
    }

    if (rank == 0) {
        return false;
    }

    uint64_t tailDim = outShape[rank - 1];
    if (tailDim == 0 || outStart % tailDim != 0) {
        return false;
    }

    bool bothTailContiguous = xStride[rank - 1] != 0 && yStride[rank - 1] != 0;
    return bothTailContiguous && (tailDim * sizeof(T) % DATABLOCK_BYTES == 0);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::BuildCopyParams(uint32_t count,
                                                                            DataCopyParams& copyParams) const
{
    copyParams.blockCount = 1;
    copyParams.blockLen = count * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::FillLocal(LocalTensor<T> dst, T value, uint32_t count)
{
    if constexpr (IsVectorShiftType<T>::value) {
        Duplicate(dst, value, count);
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            dst(i) = value;
        }
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CopyContiguousIn(LocalTensor<T> dst, GlobalTensor<T>& src,
                                                                             uint64_t offset, uint32_t count)
{
    if (count == 0) {
        return;
    }

    DataCopyParams copyParams;
    BuildCopyParams(count, copyParams);
    DataCopyPad(dst, src[offset], copyParams, {false, 0, 0, 0});
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CopyContiguousOut(GlobalTensor<T>& dst, uint64_t offset,
                                                                              LocalTensor<T> src, uint32_t count)
{
    if (count == 0) {
        return;
    }

    DataCopyParams copyParams;
    BuildCopyParams(count, copyParams);
    DataCopyPad(dst[offset], src, copyParams);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::BroadcastScalarToLocal(LocalTensor<T> dst,
                                                                                   GlobalTensor<T>& src,
                                                                                   uint64_t srcOffset, uint32_t length,
                                                                                   uint32_t fillLength)
{
    if (length == 0) {
        return;
    }

    LocalTensor<T> scalarLocal = singleBuf.Get<T>();
    DataCopyParams copyParams;
    BuildCopyParams(1, copyParams);
    DataCopyPad(scalarLocal, src[srcOffset], copyParams, {false, 0, 0, 0});
    PipeBarrier<PIPE_ALL>();
    FillLocal(dst, scalarLocal.GetValue(0), fillLength);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CopyBroadcastSegmentByOffset(
    LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint64_t tailStride, uint32_t length,
    uint32_t fillLength)
{
    if (tailStride == 0) {
        BroadcastScalarToLocal(dst, src, srcOffset, length, fillLength);
        return;
    }
    CopyContiguousIn(dst, src, srcOffset, length);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CopyBroadcastSegment(LocalTensor<T> dst,
                                                                                 GlobalTensor<T>& src,
                                                                                 const uint64_t* stride,
                                                                                 uint64_t outStart, uint32_t length,
                                                                                 uint32_t fillLength)
{
    uint64_t srcOffset = CalcOffset(outStart, stride);
    uint64_t tailStride = rank == 0 ? 0 : stride[rank - 1];
    CopyBroadcastSegmentByOffset(dst, src, srcOffset, tailStride, length, fillLength);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::BroadcastRepeatedRowsToLocal(
    LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t rowLength, uint32_t rowCount)
{
    if (rowLength == 0 || rowCount == 0) {
        return;
    }

    LocalTensor<T> rowLocal = rowBuf.Get<T>();
    CopyContiguousIn(rowLocal, src, srcOffset, rowLength);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < rowCount; ++i) {
        DataCopy(dst[i * rowLength], rowLocal, rowLength);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CopyBroadcastInputs(LocalTensor<T> xLocal,
                                                                                LocalTensor<T> yLocal,
                                                                                uint64_t outStart, uint32_t validLength,
                                                                                uint32_t computeLength)
{
    if constexpr (BROADCAST_MODE == RIGHT_SHIFT_MODE_CONTIGUOUS) {
        CopyContiguousIn(xLocal, xGm, outStart, validLength);
        CopyContiguousIn(yLocal, yGm, outStart, validLength);
        return;
    }

    if (xContiguous) {
        CopyContiguousIn(xLocal, xGm, outStart, validLength);
    } else {
        CopyBroadcastSegment(xLocal, xGm, xStride, outStart, validLength, computeLength);
    }

    if (yContiguous) {
        CopyContiguousIn(yLocal, yGm, outStart, validLength);
    } else {
        CopyBroadcastSegment(yLocal, yGm, yStride, outStart, validLength, computeLength);
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CopyBroadcastInputsPacked(
    LocalTensor<T> xLocal, LocalTensor<T> yLocal, uint64_t outStart, uint32_t validLength, uint32_t computeLength)
{
    (void)computeLength;
    uint32_t tailDim = static_cast<uint32_t>(outShape[rank - 1]);
    uint32_t rowCount = tailDim == 0 ? 0 : validLength / tailDim;
    uint64_t xOffset = CalcOffset(outStart, xStride);
    uint64_t yOffset = CalcOffset(outStart, yStride);
    uint64_t xTailStride = rank == 0 ? 0 : xStride[rank - 1];
    uint64_t yTailStride = rank == 0 ? 0 : yStride[rank - 1];

    bool xOuterBroadcast = IsOuterBroadcast(xStride, yStride);
    bool yOuterBroadcast = IsOuterBroadcast(yStride, xStride);
    if (useRowBroadcastBuffer && rowCount > 1 && xOuterBroadcast) {
        BroadcastRepeatedRowsToLocal(xLocal, xGm, xOffset, tailDim, rowCount);
        CopyContiguousIn(yLocal, yGm, yOffset, validLength);
        return;
    }
    if (useRowBroadcastBuffer && rowCount > 1 && yOuterBroadcast) {
        CopyContiguousIn(xLocal, xGm, xOffset, validLength);
        BroadcastRepeatedRowsToLocal(yLocal, yGm, yOffset, tailDim, rowCount);
        return;
    }

    uint64_t coord[RIGHT_SHIFT_MAX_BROADCAST_DIM] = {};
    InitOutputCoord(outStart, coord);
    uint32_t copiedLength = 0;
    while (copiedLength < validLength) {
        uint32_t remainLength = validLength - copiedLength;
        uint32_t segmentLength = tailDim < remainLength ? tailDim : remainLength;
        if (segmentLength == 0) {
            return;
        }

        CopyBroadcastSegmentByOffset(xLocal[copiedLength], xGm, xOffset, xTailStride, segmentLength, segmentLength);
        CopyBroadcastSegmentByOffset(yLocal[copiedLength], yGm, yOffset, yTailStride, segmentLength, segmentLength);
        copiedLength += segmentLength;
        if (copiedLength < validLength) {
            AdvanceTailSegment(coord, xOffset, yOffset);
        }
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline T RightShiftKernel<T, BROADCAST_MODE>::ReadScalarY()
{
    LocalTensor<T> yLocal = yBuf.Get<T>();
    CopyContiguousIn(yLocal, yGm, 0, 1);
    PipeBarrier<PIPE_ALL>();
    return yLocal.GetValue(0);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::InitInvalidShiftResult(LocalTensor<T> dst,
                                                                                   LocalTensor<T> src, uint32_t count)
{
    if constexpr (std::is_signed<T>::value && IsVectorShiftType<T>::value) {
        LocalTensor<T> workLocal = workBuf.Get<T>();
        Maxs(workLocal, src, static_cast<T>(-1), count);
        PipeBarrier<PIPE_ALL>();
        Mins(dst, workLocal, static_cast<T>(0), count);
    } else if constexpr (!std::is_signed<T>::value && IsVectorShiftType<T>::value) {
        Duplicate(dst, static_cast<T>(0), count);
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            dst(i) = InvalidShiftValue(src(i));
        }
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CastShiftToCompare(LocalTensor<T> ySrc,
                                                                               LocalTensor<half> yCmp, uint32_t count)
{
    if constexpr (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
        Cast(yCmp, ySrc.template ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count);
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::CompareHalfShift(LocalTensor<uint8_t> maskLocal,
                                                                             LocalTensor<half> yCmp, uint32_t shift,
                                                                             uint32_t count)
{
    switch (shift) {
        case 0:
            CompareScalar(maskLocal, yCmp, static_cast<half>(0.0), CMPMODE::EQ, count);
            break;
        case 1:
            CompareScalar(maskLocal, yCmp, static_cast<half>(1.0), CMPMODE::EQ, count);
            break;
        case 2:
            CompareScalar(maskLocal, yCmp, static_cast<half>(2.0), CMPMODE::EQ, count);
            break;
        case 3:
            CompareScalar(maskLocal, yCmp, static_cast<half>(3.0), CMPMODE::EQ, count);
            break;
        case 4:
            CompareScalar(maskLocal, yCmp, static_cast<half>(4.0), CMPMODE::EQ, count);
            break;
        case 5:
            CompareScalar(maskLocal, yCmp, static_cast<half>(5.0), CMPMODE::EQ, count);
            break;
        case 6:
            CompareScalar(maskLocal, yCmp, static_cast<half>(6.0), CMPMODE::EQ, count);
            break;
        case 7:
            CompareScalar(maskLocal, yCmp, static_cast<half>(7.0), CMPMODE::EQ, count);
            break;
        case 8:
            CompareScalar(maskLocal, yCmp, static_cast<half>(8.0), CMPMODE::EQ, count);
            break;
        case 9:
            CompareScalar(maskLocal, yCmp, static_cast<half>(9.0), CMPMODE::EQ, count);
            break;
        case 10:
            CompareScalar(maskLocal, yCmp, static_cast<half>(10.0), CMPMODE::EQ, count);
            break;
        case 11:
            CompareScalar(maskLocal, yCmp, static_cast<half>(11.0), CMPMODE::EQ, count);
            break;
        case 12:
            CompareScalar(maskLocal, yCmp, static_cast<half>(12.0), CMPMODE::EQ, count);
            break;
        case 13:
            CompareScalar(maskLocal, yCmp, static_cast<half>(13.0), CMPMODE::EQ, count);
            break;
        case 14:
            CompareScalar(maskLocal, yCmp, static_cast<half>(14.0), CMPMODE::EQ, count);
            break;
        case 15:
            CompareScalar(maskLocal, yCmp, static_cast<half>(15.0), CMPMODE::EQ, count);
            break;
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::ApplyVectorBucketShift(LocalTensor<T> dst,
                                                                                   LocalTensor<T> xSrc,
                                                                                   LocalTensor<T> ySrc, uint32_t count)
{
    if constexpr (!IsVectorShiftType<T>::value) {
        ComputeElementwise(dst, xSrc, ySrc, count);
        return;
    } else {
        constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
        constexpr uint32_t bitWidth = sizeof(T) * 8;
        uint32_t computeCount = AlignUp(count, elemsPerRepeat);
        InitInvalidShiftResult(dst, xSrc, computeCount);
        PipeBarrier<PIPE_ALL>();

        if constexpr (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
            LocalTensor<half> yCmp = cmpBuf.Get<half>();
            CastShiftToCompare(ySrc, yCmp, computeCount);
            PipeBarrier<PIPE_ALL>();
        }

        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        LocalTensor<uint64_t> maskWords = maskLocal.template ReinterpretCast<uint64_t>();
        UnaryRepeatParams shiftParams;
        shiftParams.dstBlkStride = 1;
        shiftParams.srcBlkStride = 1;
        shiftParams.dstRepStride = 8;
        shiftParams.srcRepStride = 8;

        for (uint32_t base = 0; base < computeCount; base += elemsPerRepeat) {
            for (uint32_t shift = 0; shift < bitWidth; ++shift) {
                if constexpr (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
                    LocalTensor<half> yCmp = cmpBuf.Get<half>();
                    CompareHalfShift(maskLocal, yCmp[base], shift, elemsPerRepeat);
                } else if constexpr (std::is_same<T, int32_t>::value) {
                    CompareScalar(maskLocal, ySrc[base], static_cast<int32_t>(shift), CMPMODE::EQ, elemsPerRepeat);
                } else {
                    CompareScalar(maskLocal, ySrc.template ReinterpretCast<int32_t>()[base],
                                  static_cast<int32_t>(shift), CMPMODE::EQ, elemsPerRepeat);
                }
                PipeBarrier<PIPE_ALL>();
                if constexpr (sizeof(T) == sizeof(int16_t)) {
                    uint64_t mask[2] = {maskWords(0), maskWords(1)};
                    if (mask[0] != 0 || mask[1] != 0) {
                        ShiftRight(dst[base], xSrc[base], static_cast<T>(shift), mask, 1, shiftParams);
                    }
                } else {
                    uint64_t mask[1] = {maskWords(0)};
                    if (mask[0] != 0) {
                        ShiftRight(dst[base], xSrc[base], static_cast<T>(shift), mask, 1, shiftParams);
                    }
                }
                PipeBarrier<PIPE_ALL>();
            }
        }
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::ComputeScalarShift(LocalTensor<T> dst, LocalTensor<T> src,
                                                                               uint32_t count, T shiftValue)
{
    bool validShift = !IsInvalidShift(shiftValue);
    if constexpr (IsVectorShiftType<T>::value) {
        if (validShift) {
            ShiftRight(dst, src, static_cast<T>(NormalizeShiftBits(shiftValue)), count);
            return;
        }
        InitInvalidShiftResult(dst, src, count);
        return;
    }

    for (uint32_t i = 0; i < count; ++i) {
        T xValue = src(i);
        dst(i) = validShift ? ComputeOne(xValue, shiftValue) : InvalidShiftValue(xValue);
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::ComputeElementwise(LocalTensor<T> dst, LocalTensor<T> xSrc,
                                                                               LocalTensor<T> ySrc, uint32_t count)
{
    if constexpr (IsVectorShiftType<T>::value) {
        ApplyVectorBucketShift(dst, xSrc, ySrc, count);
    } else {
        for (uint32_t i = 0; i < count; ++i) {
            dst(i) = ComputeOne(xSrc(i), ySrc(i));
        }
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::ProcessScalarY()
{
    T shiftValue = ReadScalarY();
    uint64_t processed = 0;
    while (processed < blockLength) {
        uint64_t remain = blockLength - processed;
        uint32_t count = static_cast<uint32_t>(remain > tileBufferLen ? tileBufferLen : remain);
        uint64_t start = blockOffset + processed;
        LocalTensor<T> xLocal = xBuf.Get<T>();
        LocalTensor<T> zLocal = zBuf.Get<T>();
        if (xContiguous) {
            CopyContiguousIn(xLocal, xGm, start, count);
        } else {
            CopyBroadcastSegment(xLocal, xGm, xStride, start, count, count);
        }
        PipeBarrier<PIPE_ALL>();
        ComputeScalarShift(zLocal, xLocal, count, shiftValue);
        PipeBarrier<PIPE_ALL>();
        CopyContiguousOut(zGm, start, zLocal, count);
        PipeBarrier<PIPE_ALL>();
        processed += count;
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::ProcessElementwise()
{
    uint64_t outStart = blockOffset;
    uint64_t remainLength = blockLength;
    while (remainLength > 0) {
        bool usePackedBroadcast = CanUsePackedBroadcast(outStart);
        uint32_t count = usePackedBroadcast ? CalcPackedBroadcastLength(remainLength) : 0;
        if (count == 0) {
            usePackedBroadcast = false;
            if constexpr (BROADCAST_MODE == RIGHT_SHIFT_MODE_CONTIGUOUS ||
                          BROADCAST_MODE == RIGHT_SHIFT_MODE_X_SCALAR) {
                count = static_cast<uint32_t>(remainLength > tileBufferLen ? tileBufferLen : remainLength);
            } else {
                count = CalcBroadcastSegmentLength(outStart, remainLength);
            }
        }
        if (count == 0) {
            return;
        }

        LocalTensor<T> xLocal = xBuf.Get<T>();
        LocalTensor<T> yLocal = yBuf.Get<T>();
        LocalTensor<T> zLocal = zBuf.Get<T>();
        uint32_t computeLength = count;
        if (usePackedBroadcast) {
            CopyBroadcastInputsPacked(xLocal, yLocal, outStart, count, computeLength);
        } else {
            CopyBroadcastInputs(xLocal, yLocal, outStart, count, computeLength);
        }

        PipeBarrier<PIPE_ALL>();
        ComputeElementwise(zLocal, xLocal, yLocal, count);
        PipeBarrier<PIPE_ALL>();
        CopyContiguousOut(zGm, outStart, zLocal, count);
        PipeBarrier<PIPE_ALL>();
        outStart += count;
        remainLength -= count;
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void RightShiftKernel<T, BROADCAST_MODE>::Process()
{
    if (blockLength == 0 || totalLength == 0 || tileBufferLen == 0) {
        return;
    }

    if constexpr (BROADCAST_MODE == RIGHT_SHIFT_MODE_Y_SCALAR) {
        ProcessScalarY();
        return;
    }
    ProcessElementwise();
}

template <uint32_t BROADCAST_MODE, uint32_t DTYPE_MODE>
__aicore__ inline void RightShiftKernelImpl(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                            const RightShiftTilingData* tilingData)
{
    (void)workspace;
    if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_INT8) {
        RightShiftKernel<int8_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_UINT8) {
        RightShiftKernel<uint8_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_INT16) {
        RightShiftKernel<int16_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_UINT16) {
        RightShiftKernel<uint16_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_INT32) {
        RightShiftKernel<int32_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_UINT32) {
        RightShiftKernel<uint32_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_INT64) {
        RightShiftKernel<int64_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else if constexpr (DTYPE_MODE == RIGHT_SHIFT_TPL_UINT64) {
        RightShiftKernel<uint64_t, BROADCAST_MODE> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    }
}
} // namespace NsRightShift

#endif // RIGHT_SHIFT_H
