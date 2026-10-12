/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_KERNEL_COMMON_H_
#define DETAIL_GCD_KERNEL_COMMON_H_

constexpr int64_t GCD_OUTPUT_WORD_ALIGNMENT = 16;
constexpr int64_t GCD_INT16_VECTOR_TILE = 4096;
constexpr int64_t GCD_INT16_VECTOR_MIN_ELEMENTS = 65536;
constexpr int64_t GCD_INT16_VECTOR_ALIGN_ELEMENTS = 16;
constexpr int32_t GCD_INT16_VECTOR_MAX_ITER = 22;
constexpr int64_t GCD_FP16_VECTOR_TILE = 4096;
constexpr int64_t GCD_FP16_VECTOR_MIN_ELEMENTS = 256;
constexpr int64_t GCD_FP16_VECTOR_ALIGN_ELEMENTS = 16;
constexpr int32_t GCD_FP16_VECTOR_MAX_ITER = 26;
constexpr int64_t GCD_UINT8_VECTOR_TILE = 4096;
constexpr int64_t GCD_UINT8_VECTOR_MIN_ELEMENTS = 32768;
constexpr int64_t GCD_UINT8_VECTOR_ALIGN_ELEMENTS = 32;
constexpr int32_t GCD_UINT8_VECTOR_MAX_ITER = 16;
constexpr int64_t GCD_BF16_VECTOR_TILE = 4096;
constexpr int64_t GCD_BF16_VECTOR_MIN_ELEMENTS = 512;
constexpr int64_t GCD_BF16_VECTOR_ALIGN_ELEMENTS = 16;
constexpr int32_t GCD_BF16_VECTOR_MAX_ITER = 26;
constexpr int64_t GCD_FLOAT_VECTOR_TILE = 4096;
constexpr int64_t GCD_FLOAT_VECTOR_MIN_ELEMENTS = 32768;
constexpr int64_t GCD_FLOAT_VECTOR_ALIGN_ELEMENTS = 8;
constexpr int32_t GCD_FLOAT_VECTOR_MAX_ITER = 26;
constexpr int64_t GCD_INT32_VECTOR_TILE = 4096;
constexpr int64_t GCD_INT32_VECTOR_MIN_ELEMENTS = 65536;
constexpr int64_t GCD_INT32_VECTOR_ALIGN_ELEMENTS = 8;
constexpr int32_t GCD_INT32_VECTOR_MAX_ITER = 32;
constexpr float GCD_FLOAT_VECTOR_SAFE_ABS_F = 65536.0f;
constexpr float GCD_INT32_VECTOR_SAFE_ABS_F = 1048576.0f;

__aicore__ inline int64_t CeilDivInt64(int64_t value, int64_t divisor)
{
    if (divisor <= 0) {
        return 0;
    }
    const int64_t safeDivisor = divisor;
    return (value + safeDivisor - 1) / safeDivisor;
}

__aicore__ inline int64_t AlignUpInt64(int64_t value, int64_t alignment)
{
    return CeilDivInt64(value, alignment) * alignment;
}

__aicore__ inline int64_t AlignDownInt64(int64_t value, int64_t alignment)
{
    if (alignment <= 0) {
        return 0;
    }
    int64_t checkedAlignment = alignment;
    return (value / checkedAlignment) * checkedAlignment;
}

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs) { return lhs < rhs ? lhs : rhs; }

__aicore__ inline bool NeedScalarTailSync(int64_t linear, int64_t count, int64_t vectorEnd, int64_t elementEnd)
{
    return (linear + count == vectorEnd) && (vectorEnd < elementEnd);
}

struct GcdVectorWorkBuffers {
    TPipe pipe;
    TBuf<TPosition::VECCALC> x1Buf;
    TBuf<TPosition::VECCALC> x2Buf;
    TBuf<TPosition::VECCALC> yBuf;
    TBuf<TPosition::VECCALC> aBuf;
    TBuf<TPosition::VECCALC> bBuf;
    TBuf<TPosition::VECCALC> t1Buf;
    TBuf<TPosition::VECCALC> t2Buf;
    TBuf<TPosition::VECCALC> t3Buf;
};

struct GcdByteVectorWorkBuffers {
    GcdVectorWorkBuffers base;
    TBuf<TPosition::VECCALC> x1HalfBuf;
    TBuf<TPosition::VECCALC> x2HalfBuf;
    TBuf<TPosition::VECCALC> yHalfBuf;
};

template <typename T>
struct GcdTypedTensorSet {
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;
};

__aicore__ inline void InitGcdVectorWorkBuffers(GcdVectorWorkBuffers& buffers, int64_t tile, uint64_t xBytes,
                                                uint64_t yBytes)
{
    buffers.pipe.InitBuffer(buffers.x1Buf, static_cast<uint64_t>(tile) * xBytes);
    buffers.pipe.InitBuffer(buffers.x2Buf, static_cast<uint64_t>(tile) * xBytes);
    buffers.pipe.InitBuffer(buffers.yBuf, static_cast<uint64_t>(tile) * yBytes);
    buffers.pipe.InitBuffer(buffers.aBuf, static_cast<uint64_t>(tile) * sizeof(float));
    buffers.pipe.InitBuffer(buffers.bBuf, static_cast<uint64_t>(tile) * sizeof(float));
    buffers.pipe.InitBuffer(buffers.t1Buf, static_cast<uint64_t>(tile) * sizeof(float));
    buffers.pipe.InitBuffer(buffers.t2Buf, static_cast<uint64_t>(tile) * sizeof(float));
    buffers.pipe.InitBuffer(buffers.t3Buf, static_cast<uint64_t>(tile) * sizeof(float));
}

__aicore__ inline void InitGcdByteVectorWorkBuffers(GcdByteVectorWorkBuffers& buffers, int64_t tile, uint64_t xBytes)
{
    InitGcdVectorWorkBuffers(buffers.base, tile, xBytes, sizeof(uint8_t));
    buffers.base.pipe.InitBuffer(buffers.x1HalfBuf, static_cast<uint64_t>(tile) * sizeof(half));
    buffers.base.pipe.InitBuffer(buffers.x2HalfBuf, static_cast<uint64_t>(tile) * sizeof(half));
    buffers.base.pipe.InitBuffer(buffers.yHalfBuf, static_cast<uint64_t>(tile) * sizeof(half));
}

template <HardEvent EVENT>
__aicore__ inline void SyncPipe()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(EVENT));
    SetFlag<EVENT>(eventId);
    WaitFlag<EVENT>(eventId);
}

__aicore__ inline void SyncMte3ToScalarIf(bool enabled)
{
    if (enabled) {
        SyncPipe<HardEvent::MTE3_S>();
    }
}

__aicore__ inline uint8_t ReadPackedByte(GlobalTensor<uint32_t>& gm, int64_t elementOffset)
{
    uint32_t word = gm.GetValue(static_cast<uint64_t>(elementOffset >> 2));
    return static_cast<uint8_t>((word >> ((elementOffset & 3) * 8)) & 0xffU);
}

__aicore__ inline uint16_t ReadPackedHalf(GlobalTensor<uint32_t>& gm, int64_t elementOffset)
{
    uint32_t word = gm.GetValue(static_cast<uint64_t>(elementOffset >> 1));
    return static_cast<uint16_t>((word >> ((elementOffset & 1) * 16)) & 0xffffU);
}

__aicore__ inline void RunFloatEuclid(LocalTensor<float> aLocal, LocalTensor<float> bLocal, LocalTensor<float> t1,
                                      LocalTensor<float> t2, LocalTensor<float> t3, int32_t count, int32_t maxIter)
{
    for (int32_t iter = 0; iter < maxIter; ++iter) {
        Maxs(t1, bLocal, 1.0f, count);
        Div(t1, aLocal, t1, count);
        Cast(t2.ReinterpretCast<int32_t>(), t1, RoundMode::CAST_FLOOR, count);
        Cast(t1, t2.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
        Mul(t1, t1, bLocal, count);
        Sub(t1, aLocal, t1, count);
        Maxs(t1, t1, 0.0f, count);
        Mins(t2, bLocal, 1.0f, count);
        Sub(t3, bLocal, aLocal, count);
        Mul(bLocal, t1, t2, count);
        Mul(t3, t3, t2, count);
        Add(aLocal, aLocal, t3, count);
    }
}

__aicore__ inline void FloorFloatVectorInputs(LocalTensor<float> aLocal, LocalTensor<float> bLocal,
                                              LocalTensor<float> tempLocal, int32_t count)
{
    Cast(tempLocal.ReinterpretCast<int32_t>(), aLocal, RoundMode::CAST_FLOOR, count);
    Cast(aLocal, tempLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
    Cast(tempLocal.ReinterpretCast<int32_t>(), bLocal, RoundMode::CAST_FLOOR, count);
    Cast(bLocal, tempLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
}

template <typename ByteT, bool ABS_INPUTS>
__aicore__ inline void ComputeByteVectorTile(GcdByteVectorWorkBuffers& buffers, GlobalTensor<ByteT>& x1ByteGm,
                                             GlobalTensor<ByteT>& x2ByteGm, GlobalTensor<uint8_t>& yByteGm,
                                             int64_t linear, int32_t count, int32_t maxIter, bool syncBeforeScalarTail)
{
    LocalTensor<ByteT> x1Local = buffers.base.x1Buf.Get<ByteT>();
    LocalTensor<ByteT> x2Local = buffers.base.x2Buf.Get<ByteT>();
    LocalTensor<uint8_t> yLocal = buffers.base.yBuf.Get<uint8_t>();
    LocalTensor<half> x1Half = buffers.x1HalfBuf.Get<half>();
    LocalTensor<half> x2Half = buffers.x2HalfBuf.Get<half>();
    LocalTensor<half> yHalf = buffers.yHalfBuf.Get<half>();
    LocalTensor<float> aLocal = buffers.base.aBuf.Get<float>();
    LocalTensor<float> bLocal = buffers.base.bBuf.Get<float>();
    LocalTensor<float> t1 = buffers.base.t1Buf.Get<float>();
    LocalTensor<float> t2 = buffers.base.t2Buf.Get<float>();
    LocalTensor<float> t3 = buffers.base.t3Buf.Get<float>();

    DataCopy(x1Local, x1ByteGm[static_cast<uint64_t>(linear)], count);
    DataCopy(x2Local, x2ByteGm[static_cast<uint64_t>(linear)], count);
    SyncPipe<HardEvent::MTE2_V>();
    Cast(x1Half, x1Local, RoundMode::CAST_NONE, count);
    Cast(x2Half, x2Local, RoundMode::CAST_NONE, count);
    Cast(aLocal, x1Half, RoundMode::CAST_NONE, count);
    Cast(bLocal, x2Half, RoundMode::CAST_NONE, count);
    if constexpr (ABS_INPUTS) {
        Abs(aLocal, aLocal, count);
        Abs(bLocal, bLocal, count);
    }

    RunFloatEuclid(aLocal, bLocal, t1, t2, t3, count, maxIter);

    Cast(yHalf, aLocal, RoundMode::CAST_RINT, count);
    Cast(yLocal, yHalf, RoundMode::CAST_RINT, count);
    SyncPipe<HardEvent::V_MTE3>();
    DataCopy(yByteGm[static_cast<uint64_t>(linear)], yLocal, count);
    SyncMte3ToScalarIf(syncBeforeScalarTail);
}

__aicore__ inline void RunInt32Euclid(LocalTensor<float> aLocal, LocalTensor<float> bLocal, LocalTensor<float> t1,
                                      LocalTensor<float> t2, LocalTensor<float> t3, LocalTensor<uint8_t> cmpMask,
                                      int32_t count)
{
    for (int32_t iter = 0; iter < GCD_INT32_VECTOR_MAX_ITER; ++iter) {
        Maxs(t1, bLocal, 1.0f, count);
        Div(t1, aLocal, t1, count);
        Cast(t2.ReinterpretCast<int32_t>(), t1, RoundMode::CAST_FLOOR, count);
        Cast(t1, t2.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
        Mul(t1, t1, bLocal, count);
        Sub(t1, aLocal, t1, count);
        Add(t3, t1, bLocal, count);
        Compares(cmpMask, t1, 0.0f, CMPMODE::LT, static_cast<uint32_t>(count));
        Select(t1, cmpMask, t3, t1, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Sub(t3, t1, bLocal, count);
        Compare(cmpMask, t1, bLocal, CMPMODE::GE, static_cast<uint32_t>(count));
        Select(t1, cmpMask, t3, t1, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(count));
        Maxs(t1, t1, 0.0f, count);
        Mins(t2, bLocal, 1.0f, count);
        Sub(t3, bLocal, aLocal, count);
        Mul(bLocal, t1, t2, count);
        Mul(t3, t3, t2, count);
        Add(aLocal, aLocal, t3, count);
    }
}

#endif // DETAIL_GCD_KERNEL_COMMON_H_
