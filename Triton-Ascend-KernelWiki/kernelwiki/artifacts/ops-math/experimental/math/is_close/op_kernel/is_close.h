/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IS_CLOSE_H
#define IS_CLOSE_H

#include <limits>
#include <type_traits>

#include "is_close_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "is_close_tiling_key.h"
#endif
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NsIsClose {
using namespace AscendC;
using BF16 = bfloat16_t;
using FP16 = half;
using INT32 = int32_t;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SCALAR_BUFFER_NUM = 1;
constexpr uint32_t COMPARE_ALIGN = 256;
constexpr uint32_t DATABLOCK_BYTES = 32;
constexpr uint32_t BROADCAST_SCALAR_BUFFER_BYTES = 32;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

template <typename T, uint32_t BROADCAST_MODE>
class IsClose {
public:
    __aicore__ inline IsClose() {}
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const IsCloseTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t progress, uint32_t validLength, uint32_t computeLength);
    __aicore__ inline void CopyBroadcastInput(uint64_t outStart, uint32_t validLength, uint32_t computeLength);
    __aicore__ inline void CopyBroadcastInputPacked(uint64_t outStart, uint32_t validLength, uint32_t computeLength);
    __aicore__ inline void Compute(uint32_t computeLength);
    __aicore__ inline void CopyOut(uint64_t outOffset, uint32_t validLength);
    __aicore__ inline void CastInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t computeLength);
    __aicore__ inline uint64_t CalcInputOffset(uint64_t outIndex, const uint64_t* stride);
    __aicore__ inline void InitOutputCoord(uint64_t outIndex, uint64_t* coord);
    __aicore__ inline void AdvanceTailSegment(uint64_t* coord, uint64_t& x1Offset, uint64_t& x2Offset);
    __aicore__ inline uint32_t CalcBroadcastSegmentLength(uint64_t outStart, uint64_t remainLength);
    __aicore__ inline uint32_t CalcPackedBroadcastLength(uint64_t remainLength);
    __aicore__ inline bool CanUsePackedBroadcast(uint64_t outStart);
    __aicore__ inline bool IsOuterRowBroadcast(const uint64_t* stride);
    __aicore__ inline bool IsOutputContiguousStride(const uint64_t* stride);
    __aicore__ inline bool IsOuterBroadcast(const uint64_t* broadcastStride, const uint64_t* contiguousStride);
    __aicore__ inline void BroadcastOneValue(LocalTensor<T> dst, LocalTensor<T> src, uint32_t length);
    __aicore__ inline void CopyContiguousToLocal(
        LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t length);
    __aicore__ inline void BroadcastRepeatedRowsToLocal(
        LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t rowLength, uint32_t rowCount);
    __aicore__ inline void BroadcastScalarToLocal(
        LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t length, uint32_t fillLength,
        TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM>& scalarQueue);
    __aicore__ inline void CopyBroadcastSegment(
        LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint64_t tailStride, uint32_t length,
        uint32_t fillLength, TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM>& scalarQueue);
    __aicore__ inline void ProcessBroadcast();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM> x1ScalarQueue;
    TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM> x2ScalarQueue;
    TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM> broadcastRowQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<QuePosition::VECCALC> x1FloatBuf;
    TBuf<QuePosition::VECCALC> x2FloatBuf;
    TBuf<QuePosition::VECCALC> actualErrorBuf;
    TBuf<QuePosition::VECCALC> allowedErrorBuf;
    TBuf<QuePosition::VECCALC> resultBuf;
    TBuf<QuePosition::VECCALC> resultHalfBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> infBuf;
    TBuf<QuePosition::VECCALC> maskBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<int8_t> yGm;

    uint64_t blockLength = 0;
    uint64_t blockOffset = 0;
    uint64_t loopCount = 0;
    uint64_t tileBufferLen = 0;
    uint64_t tailTileLen = 0;
    float rtol = 1e-5f;
    float atol = 1e-8f;
    uint32_t equalNan = 0;
    uint32_t rank = 0;
    uint64_t outShape[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    uint64_t x1Stride[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    uint64_t x2Stride[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    bool useRowBroadcastBuffer = false;
    float infValue = std::numeric_limits<float>::infinity();
};

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const IsCloseTilingData* tilingData)
{
    (void)workspace;
    uint64_t blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData->formerCoreNum) {
        blockLength = tilingData->tailCoreDataNum;
        loopCount = tilingData->tailCoreLoopCount;
        tailTileLen = tilingData->tailCoreTailDataNum;
        blockOffset = tilingData->formerCoreNum * tilingData->formerCoreDataNum +
                      (blockIdx - tilingData->formerCoreNum) * blockLength;
    } else {
        blockLength = tilingData->formerCoreDataNum;
        loopCount = tilingData->formerCoreLoopCount;
        tailTileLen = tilingData->formerCoreTailDataNum;
        blockOffset = blockLength * blockIdx;
    }

    tileBufferLen = tilingData->tileBufferLen;
    rtol = tilingData->rtol;
    atol = tilingData->atol;
    equalNan = tilingData->equalNan;
    rank = tilingData->rank;
    for (uint32_t i = 0; i < IS_CLOSE_MAX_BROADCAST_DIM; ++i) {
        outShape[i] = tilingData->outShape[i];
        x1Stride[i] = tilingData->x1Stride[i];
        x2Stride[i] = tilingData->x2Stride[i];
    }

    x1Gm.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2);
    yGm.SetGlobalBuffer((__gm__ int8_t*)y);

    if (tileBufferLen == 0) {
        return;
    }

    uint64_t broadcastRowBufferBytes = BROADCAST_SCALAR_BUFFER_BYTES;
    useRowBroadcastBuffer = false;
    if (rank > 1) {
        uint64_t tailDim = outShape[rank - 1];
        bool rowBytesAligned = tailDim > 0 && (tailDim * sizeof(T) % DATABLOCK_BYTES) == 0;
        bool x1OuterBroadcast = IsOuterBroadcast(x1Stride, x2Stride);
        bool x2OuterBroadcast = IsOuterBroadcast(x2Stride, x1Stride);
        useRowBroadcastBuffer = tailDim > 0 && tailDim <= tileBufferLen && rowBytesAligned &&
                                (x1OuterBroadcast || x2OuterBroadcast);
        if (useRowBroadcastBuffer) {
            broadcastRowBufferBytes = tailDim * sizeof(T);
        }
    }

    pipe.InitBuffer(x1Queue, BUFFER_NUM, tileBufferLen * sizeof(T));
    pipe.InitBuffer(x2Queue, BUFFER_NUM, tileBufferLen * sizeof(T));
    pipe.InitBuffer(x1ScalarQueue, SCALAR_BUFFER_NUM, BROADCAST_SCALAR_BUFFER_BYTES);
    pipe.InitBuffer(x2ScalarQueue, SCALAR_BUFFER_NUM, BROADCAST_SCALAR_BUFFER_BYTES);
    pipe.InitBuffer(broadcastRowQueue, SCALAR_BUFFER_NUM, broadcastRowBufferBytes);
    pipe.InitBuffer(yQueue, BUFFER_NUM, tileBufferLen * sizeof(int8_t));
    pipe.InitBuffer(x1FloatBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(x2FloatBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(actualErrorBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(allowedErrorBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(resultBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(resultHalfBuf, tileBufferLen * sizeof(half));
    pipe.InitBuffer(onesBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(zerosBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(infBuf, tileBufferLen * sizeof(float));
    pipe.InitBuffer(maskBuf, tileBufferLen * sizeof(uint8_t));
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint64_t IsClose<T, BROADCAST_MODE>::CalcInputOffset(uint64_t outIndex, const uint64_t* stride)
{
    uint64_t inputOffset = 0;
    uint64_t remain = outIndex;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        uint64_t dim = outShape[static_cast<uint32_t>(i)];
        uint64_t coord = dim == 0 ? 0 : remain % dim;
        remain = dim == 0 ? 0 : remain / dim;
        inputOffset += coord * stride[static_cast<uint32_t>(i)];
    }
    return inputOffset;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::InitOutputCoord(uint64_t outIndex, uint64_t* coord)
{
    uint64_t remain = outIndex;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        uint32_t dimIndex = static_cast<uint32_t>(i);
        uint64_t dim = outShape[dimIndex];
        coord[dimIndex] = dim == 0 ? 0 : remain % dim;
        remain = dim == 0 ? 0 : remain / dim;
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::AdvanceTailSegment(
    uint64_t* coord, uint64_t& x1Offset, uint64_t& x2Offset)
{
    if (rank <= 1) {
        return;
    }

    for (int32_t i = static_cast<int32_t>(rank) - 2; i >= 0; --i) {
        uint32_t dimIndex = static_cast<uint32_t>(i);
        coord[dimIndex]++;
        x1Offset += x1Stride[dimIndex];
        x2Offset += x2Stride[dimIndex];
        if (coord[dimIndex] < outShape[dimIndex]) {
            return;
        }

        coord[dimIndex] = 0;
        x1Offset -= outShape[dimIndex] * x1Stride[dimIndex];
        x2Offset -= outShape[dimIndex] * x2Stride[dimIndex];
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline uint32_t IsClose<T, BROADCAST_MODE>::CalcBroadcastSegmentLength(
    uint64_t outStart, uint64_t remainLength)
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
__aicore__ inline uint32_t IsClose<T, BROADCAST_MODE>::CalcPackedBroadcastLength(uint64_t remainLength)
{
    uint64_t tailDim = outShape[rank - 1];
    uint64_t maxLength = remainLength > tileBufferLen ? tileBufferLen : remainLength;
    return static_cast<uint32_t>(maxLength / tailDim * tailDim);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool IsClose<T, BROADCAST_MODE>::CanUsePackedBroadcast(uint64_t outStart)
{
    if (rank == 0) {
        return false;
    }

    uint64_t tailDim = outShape[rank - 1];
    if (tailDim == 0) {
        return false;
    }

    bool bothTailContiguous = x1Stride[rank - 1] != 0 && x2Stride[rank - 1] != 0;
    if (!bothTailContiguous) {
        return false;
    }

    return (outStart % tailDim == 0) && (tailDim * sizeof(T) % DATABLOCK_BYTES == 0);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool IsClose<T, BROADCAST_MODE>::IsOuterRowBroadcast(const uint64_t* stride)
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
__aicore__ inline bool IsClose<T, BROADCAST_MODE>::IsOutputContiguousStride(const uint64_t* stride)
{
    if (rank == 0) {
        return false;
    }

    uint64_t expectedStride = 1;
    for (int32_t i = static_cast<int32_t>(rank) - 1; i >= 0; --i) {
        uint32_t dimIndex = static_cast<uint32_t>(i);
        if (stride[dimIndex] != expectedStride) {
            return false;
        }
        expectedStride *= outShape[dimIndex];
    }
    return true;
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline bool IsClose<T, BROADCAST_MODE>::IsOuterBroadcast(
    const uint64_t* broadcastStride, const uint64_t* contiguousStride)
{
    if (!IsOuterRowBroadcast(broadcastStride)) {
        return false;
    }

    return IsOutputContiguousStride(contiguousStride);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::BroadcastOneValue(
    LocalTensor<T> dst, LocalTensor<T> src, uint32_t length)
{
    if (length == 0) {
        return;
    }

    Duplicate(dst, src.GetValue(0), length);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CopyContiguousToLocal(
    LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t length)
{
    if (length == 0) {
        return;
    }

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = length * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(dst, src[srcOffset], copyParams, {false, 0, 0, 0});
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::BroadcastRepeatedRowsToLocal(
    LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t rowLength, uint32_t rowCount)
{
    if (rowLength == 0 || rowCount == 0) {
        return;
    }

    LocalTensor<T> srcLocal = broadcastRowQueue.AllocTensor<T>();
    CopyContiguousToLocal(srcLocal, src, srcOffset, rowLength);
    broadcastRowQueue.EnQue(srcLocal);
    srcLocal = broadcastRowQueue.DeQue<T>();
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < rowCount; ++i) {
        DataCopy(dst[i * rowLength], srcLocal, rowLength);
    }
    PipeBarrier<PIPE_ALL>();
    broadcastRowQueue.FreeTensor(srcLocal);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::BroadcastScalarToLocal(
    LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint32_t length, uint32_t fillLength,
    TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM>& scalarQueue)
{
    if (length == 0) {
        return;
    }

    LocalTensor<T> scalarLocal = scalarQueue.AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    // DataCopyPad pads this 32B block with the first copied element, which gives Brcb 8 aligned source values.
    DataCopyPad(scalarLocal, src[srcOffset], copyParams, {false, 0, 0, 0});
    scalarQueue.EnQue(scalarLocal);
    scalarLocal = scalarQueue.DeQue<T>();
    BroadcastOneValue(dst, scalarLocal, fillLength);
    scalarQueue.FreeTensor(scalarLocal);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CopyBroadcastSegment(
    LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t srcOffset, uint64_t tailStride, uint32_t length,
    uint32_t fillLength, TQue<QuePosition::VECIN, SCALAR_BUFFER_NUM>& scalarQueue)
{
    if (tailStride == 0) {
        BroadcastScalarToLocal(dst, src, srcOffset, length, fillLength, scalarQueue);
        return;
    }
    CopyContiguousToLocal(dst, src, srcOffset, length);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CopyBroadcastInput(
    uint64_t outStart, uint32_t validLength, uint32_t computeLength)
{
    LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
    LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();

    uint64_t x1Offset = CalcInputOffset(outStart, x1Stride);
    uint64_t x2Offset = CalcInputOffset(outStart, x2Stride);
    uint64_t x1TailStride = rank == 0 ? 0 : x1Stride[rank - 1];
    uint64_t x2TailStride = rank == 0 ? 0 : x2Stride[rank - 1];
    CopyBroadcastSegment(x1Local, x1Gm, x1Offset, x1TailStride, validLength, computeLength, x1ScalarQueue);
    CopyBroadcastSegment(x2Local, x2Gm, x2Offset, x2TailStride, validLength, computeLength, x2ScalarQueue);

    PipeBarrier<PIPE_V>();
    x1Queue.EnQue(x1Local);
    x2Queue.EnQue(x2Local);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CopyBroadcastInputPacked(
    uint64_t outStart, uint32_t validLength, uint32_t computeLength)
{
    LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
    LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
    uint64_t x1TailStride = rank == 0 ? 0 : x1Stride[rank - 1];
    uint64_t x2TailStride = rank == 0 ? 0 : x2Stride[rank - 1];
    uint32_t tailDim = rank == 0 ? 0 : static_cast<uint32_t>(outShape[rank - 1]);
    uint64_t coord[IS_CLOSE_MAX_BROADCAST_DIM] = {};
    InitOutputCoord(outStart, coord);
    uint64_t x1Offset = CalcInputOffset(outStart, x1Stride);
    uint64_t x2Offset = CalcInputOffset(outStart, x2Stride);
    bool rowBytesAligned = (tailDim * sizeof(T) % DATABLOCK_BYTES) == 0;
    bool canRepeatRows =
        useRowBroadcastBuffer && tailDim != 0 && tailDim <= tileBufferLen && validLength % tailDim == 0 &&
        rowBytesAligned;
    uint32_t rowCount = canRepeatRows ? validLength / tailDim : 0;
    bool x1OuterBroadcast = IsOuterBroadcast(x1Stride, x2Stride);
    bool x2OuterBroadcast = IsOuterBroadcast(x2Stride, x1Stride);
    if (rowCount > 1 && x1OuterBroadcast) {
        BroadcastRepeatedRowsToLocal(x1Local, x1Gm, x1Offset, tailDim, rowCount);
        CopyContiguousToLocal(x2Local, x2Gm, x2Offset, validLength);
        PipeBarrier<PIPE_V>();
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
        return;
    }
    if (rowCount > 1 && x2OuterBroadcast) {
        BroadcastRepeatedRowsToLocal(x2Local, x2Gm, x2Offset, tailDim, rowCount);
        CopyContiguousToLocal(x1Local, x1Gm, x1Offset, validLength);
        PipeBarrier<PIPE_V>();
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
        return;
    }
    uint32_t copiedLength = 0;
    while (copiedLength < validLength) {
        uint32_t remainLength = validLength - copiedLength;
        uint32_t segmentLength = tailDim;
        segmentLength = segmentLength < remainLength ? segmentLength : remainLength;
        if (segmentLength == 0) {
            break;
        }

        uint32_t fillLength = segmentLength;
        if (copiedLength + segmentLength == validLength) {
            fillLength += computeLength - validLength;
        }

        CopyBroadcastSegment(
            x1Local[copiedLength], x1Gm, x1Offset, x1TailStride, segmentLength, fillLength, x1ScalarQueue);
        CopyBroadcastSegment(
            x2Local[copiedLength], x2Gm, x2Offset, x2TailStride, segmentLength, fillLength, x2ScalarQueue);
        copiedLength += segmentLength;
        if (copiedLength < validLength) {
            AdvanceTailSegment(coord, x1Offset, x2Offset);
        }
    }

    PipeBarrier<PIPE_V>();
    x1Queue.EnQue(x1Local);
    x2Queue.EnQue(x2Local);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CopyIn(
    uint64_t progress, uint32_t validLength, uint32_t computeLength)
{
    (void)computeLength;
    LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
    LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();

    uint64_t outStart = blockOffset + progress * tileBufferLen;
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = validLength * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;


    DataCopyPad(x1Local, x1Gm[outStart], copyParams, {false, 0, 0, 0});
    DataCopyPad(x2Local, x2Gm[outStart], copyParams, {false, 0, 0, 0});
    x1Queue.EnQue(x1Local);
    x2Queue.EnQue(x2Local);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CastInputToFloat(
    LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t tileLength)
{
    if constexpr (std::is_same_v<T, float>) {
        (void)tileLength;
        dst = src;
    } else {
        Cast(dst, src, RoundMode::CAST_NONE, tileLength);
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::Compute(uint32_t computeLength)
{
    LocalTensor<T> x1Local = x1Queue.DeQue<T>();
    LocalTensor<T> x2Local = x2Queue.DeQue<T>();
    LocalTensor<int8_t> yLocal = yQueue.AllocTensor<int8_t>();

    LocalTensor<float> x1Float = x1FloatBuf.Get<float>();
    LocalTensor<float> x2Float = x2FloatBuf.Get<float>();
    LocalTensor<float> actualError = actualErrorBuf.Get<float>();
    LocalTensor<float> allowedError = allowedErrorBuf.Get<float>();
    LocalTensor<float> result = resultBuf.Get<float>();
    LocalTensor<half> resultHalf = resultHalfBuf.Get<half>();
    LocalTensor<float> ones = onesBuf.Get<float>();
    LocalTensor<float> zeros = zerosBuf.Get<float>();
    LocalTensor<float> infTensor = infBuf.Get<float>();
    LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();

    CastInputToFloat(x1Float, x1Local, computeLength);
    CastInputToFloat(x2Float, x2Local, computeLength);

    Sub(actualError, x1Float, x2Float, computeLength);
    Abs(actualError, actualError, computeLength);
    Abs(allowedError, x2Float, computeLength);
    Muls(allowedError, allowedError, rtol, computeLength);
    Adds(allowedError, allowedError, atol, computeLength);

    Duplicate(ones, 1.0f, computeLength);
    Duplicate(zeros, 0.0f, computeLength);
    Duplicate(infTensor, infValue, computeLength);

    Compare(mask, actualError, allowedError, CMPMODE::LE, computeLength);
    Select(result, mask, ones, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);

    Compare(mask, actualError, infTensor, CMPMODE::LT, computeLength);
    Select(result, mask, result, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);

    Compare(mask, x1Float, x2Float, CMPMODE::EQ, computeLength);
    Select(result, mask, ones, result, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);

    if (equalNan != 0U) {
        Abs(actualError, x1Float, computeLength);
        Compare(mask, actualError, infTensor, CMPMODE::LE, computeLength);
        Select(actualError, mask, zeros, ones, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);
        Abs(allowedError, x2Float, computeLength);
        Compare(mask, allowedError, infTensor, CMPMODE::LE, computeLength);
        Select(allowedError, mask, zeros, ones, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);
        Add(actualError, actualError, allowedError, computeLength);
        CompareScalar(mask, actualError, 2.0f, CMPMODE::EQ, computeLength);
        Select(result, mask, ones, result, SELMODE::VSEL_TENSOR_TENSOR_MODE, computeLength);
    }

    Cast(resultHalf, result, RoundMode::CAST_RINT, computeLength);
    Cast(yLocal, resultHalf, RoundMode::CAST_RINT, computeLength);

    x1Queue.FreeTensor(x1Local);
    x2Queue.FreeTensor(x2Local);
    yQueue.EnQue<int8_t>(yLocal);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::CopyOut(uint64_t outOffset, uint32_t validLength)
{
    LocalTensor<int8_t> yLocal = yQueue.DeQue<int8_t>();

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = validLength * sizeof(int8_t);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(yGm[outOffset], yLocal, copyParams);
    yQueue.FreeTensor(yLocal);
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::ProcessBroadcast()
{
    if (blockLength == 0 || loopCount == 0 || tileBufferLen == 0) {
        return;
    }

    uint64_t outStart = blockOffset;
    uint64_t remainLength = blockLength;
    while (remainLength > 0) {
        bool usePackedBroadcast = CanUsePackedBroadcast(outStart);
        uint32_t validLength = usePackedBroadcast ? CalcPackedBroadcastLength(remainLength) : 0;
        if (validLength == 0) {
            usePackedBroadcast = false;
            validLength = CalcBroadcastSegmentLength(outStart, remainLength);
        }
        if (validLength == 0) {
            return;
        }
        uint32_t computeLength = AlignUp(validLength, COMPARE_ALIGN);
        if (usePackedBroadcast) {
            CopyBroadcastInputPacked(outStart, validLength, computeLength);
        } else {
            CopyBroadcastInput(outStart, validLength, computeLength);
        }
        Compute(computeLength);
        CopyOut(outStart, validLength);
        outStart += validLength;
        remainLength -= validLength;
    }
}

template <typename T, uint32_t BROADCAST_MODE>
__aicore__ inline void IsClose<T, BROADCAST_MODE>::Process()
{
    if (blockLength == 0 || loopCount == 0 || tileBufferLen == 0) {
        return;
    }

    if constexpr (BROADCAST_MODE != IS_CLOSE_BROADCAST_MODE_CONTIGUOUS) {
        ProcessBroadcast();
        return;
    }

    for (uint64_t i = 0; i + 1 < loopCount; ++i) {
        uint32_t validLength = static_cast<uint32_t>(tileBufferLen);
        uint32_t computeLength = AlignUp(validLength, COMPARE_ALIGN);
        uint64_t outOffset = blockOffset + i * tileBufferLen;
        CopyIn(i, validLength, computeLength);
        Compute(computeLength);
        CopyOut(outOffset, validLength);
    }

    uint32_t validLength = static_cast<uint32_t>(tailTileLen);
    uint32_t computeLength = AlignUp(validLength, COMPARE_ALIGN);
    uint64_t outOffset = blockOffset + (loopCount - 1) * tileBufferLen;
    CopyIn(loopCount - 1, validLength, computeLength);
    Compute(computeLength);
    CopyOut(outOffset, validLength);
}

template <uint32_t BROADCAST_MODE, uint32_t IS_CLOSE_DTYPE_MODE>
__aicore__ inline void IsCloseKernelImpl(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const IsCloseTilingData* tilingData)
{
    if constexpr (IS_CLOSE_DTYPE_MODE == IS_CLOSE_TPL_FP32) {
        IsClose<float, BROADCAST_MODE> op;
        op.Init(x1, x2, y, workspace, tilingData);
        op.Process();
    } else if constexpr (IS_CLOSE_DTYPE_MODE == IS_CLOSE_TPL_FP16) {
        IsClose<FP16, BROADCAST_MODE> op;
        op.Init(x1, x2, y, workspace, tilingData);
        op.Process();
    } else if constexpr (IS_CLOSE_DTYPE_MODE == IS_CLOSE_TPL_BF16) {
        IsClose<BF16, BROADCAST_MODE> op;
        op.Init(x1, x2, y, workspace, tilingData);
        op.Process();
    } else if constexpr (IS_CLOSE_DTYPE_MODE == IS_CLOSE_TPL_INT32) {
        IsClose<INT32, BROADCAST_MODE> op;
        op.Init(x1, x2, y, workspace, tilingData);
        op.Process();
    }
}
} // namespace NsIsClose

#endif // IS_CLOSE_H
