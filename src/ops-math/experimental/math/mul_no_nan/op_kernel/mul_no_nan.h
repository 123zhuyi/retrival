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
 * \file mul_no_nan.h
 * \brief MulNoNan 910b (A2/A3) AiCore kernel.
 *        y = (x2 == 0) ? 0 : x1 * x2, element-wise with ND broadcast support.
 *
 * Vectorized implementation following the A5 (arch35) MulNoNan DAG semantics:
 *   mask = (x2 != 0); y = mask ? (x1 * x2) : 0
 * using Compares(NE) + Mul + Select(VSEL_TENSOR_SCALAR_MODE) on AIV. Unlike the
 * earlier Abs+Muls+Mins trick, Select never produces Inf*0=NaN in the zero
 * branch, so the result is correct even when x1 is Inf and x2 is 0.
 *
 * NOTE: VSEL_TENSOR_TENSOR_MODE (tensor-tensor select) is unreliable on 910B
 * AIV (bitmap mask chain), so we use VSEL_TENSOR_SCALAR_MODE (second source is
 * the scalar 0) which is the proven pattern used by the Equal operator.
 *
 * Multi-core block partition via GetBlockIdx/GetBlockNum. The flattened output
 * range of a core is walked in "contiguous segments" (largest runs where both
 * inputs are memory-contiguous given the broadcast strides). Large aligned
 * segments run fully vectorized through a double-buffered TQue pipeline
 * (DataCopy in -> vector compute -> DataCopy out). Unaligned heads/tails
 * (usually < 32B worth of elements) and small broadcast-replicated segments
 * fall back to a scalar loop, which is bounded to reach 32B alignment so the
 * bulk of a contiguous tensor stays vectorized (scalar GM GetValue/SetValue is
 * ~100x slower than the vector path and must not run over large runs).
 * fp16/bf16 inputs are promoted to fp32 for cmp/mul/select and rounded back
 * (same promotion as the A5 FloatCast DAG).
 */
#ifndef MUL_NO_NAN_H_
#define MUL_NO_NAN_H_

#include "kernel_operator.h"
#include "mul_no_nan_tiling_data.h"
#include "mul_no_nan_tiling_key.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace {

constexpr int32_t MUL_NO_NAN_TILE_ELEMS = 2048;
constexpr int32_t MUL_NO_NAN_VEC_MIN_ELEMS = 64;
constexpr int32_t MUL_NO_NAN_VEC_REPEAT = 64;

__aicore__ inline int64_t MulNoNanCeilDiv(int64_t value, int64_t divisor)
{
    if (divisor <= 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

// Partition [0, total) among the launched cores, with every core's range
// aligned to 32B (alignElems elements). This guarantees the bulk of a
// contiguous tensor is processed fully vectorized: mixing the slow scalar
// GetValue/SetValue path with the TQue vector pipeline in the same core was
// found to be unreliable on 910B AIV, so we avoid scalar heads/tails in the
// common aligned case entirely.
__aicore__ inline void MulNoNanCalcBlockRange(int64_t total, int64_t alignElems, int64_t& begin, int64_t& end)
{
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockNum <= 1) {
        begin = 0;
        end = total;
        return;
    }
    int64_t perBlock = MulNoNanCeilDiv(total, blockNum);
    // Align each core's range to a full TILE (MUL_NO_NAN_TILE_ELEMS): partial
    // vector tiles are unreliable on 910B AIV (count-based API mishandles the
    // last partial repeat), so we only feed the vector pipeline whole tiles.
    int64_t tileElems = static_cast<int64_t>(MUL_NO_NAN_TILE_ELEMS);
    perBlock = (perBlock / tileElems) * tileElems;
    if (perBlock < tileElems) {
        perBlock = tileElems;
    }
    begin = blockIdx * perBlock;
    if (begin > total) {
        begin = total;
    }
    end = (blockIdx + 1) * perBlock;
    if (blockIdx == blockNum - 1) {
        end = total; // last core takes the remainder so [0, total) is covered
    }
    if (end > total) {
        end = total;
    }
}

// product of output dims strictly inside (to the right of) dim d
__aicore__ inline int64_t MulNoNanInnerProduct(const MulNoNanTilingData& tiling, int64_t dim)
{
    int64_t ip = 1;
    for (int64_t e = dim + 1; e < tiling.rank; ++e) {
        ip *= tiling.outputDims[e];
    }
    return ip;
}

// Largest forward run (in flattened output elements) starting at `pos` for which
// the input with the given strides is memory-contiguous (stride == 1 per element).
__aicore__ inline int64_t MulNoNanInputContigRun(const MulNoNanTilingData& tiling, const int64_t* strides, int64_t pos)
{
    int64_t k0 = tiling.rank; // not even the innermost dim is contiguous
    for (int64_t d = tiling.rank - 1; d >= 0; --d) {
        if (strides[d] == MulNoNanInnerProduct(tiling, d)) {
            k0 = d;
        } else {
            break;
        }
    }
    if (k0 == tiling.rank) {
        return 1;
    }
    int64_t segLen = 1;
    for (int64_t e = k0; e < tiling.rank; ++e) {
        segLen *= tiling.outputDims[e];
    }
    return segLen - (pos % segLen);
}

// Input memory offset for flattened output position `pos`.
__aicore__ inline int64_t MulNoNanInputOffset(const MulNoNanTilingData& tiling, const int64_t* strides, int64_t pos)
{
    int64_t off = 0;
    for (int64_t d = tiling.rank - 1; d >= 0; --d) {
        int64_t dim = tiling.outputDims[d];
        off += (pos % dim) * strides[d];
        pos /= dim;
    }
    return off;
}

__aicore__ inline bool MulNoNanAligned32(int64_t elemOffset, int64_t elemSize, int64_t count)
{
    return ((elemOffset * elemSize) % 32 == 0) && ((count * elemSize) % 32 == 0);
}

// The 910b scalar unit has no bf16 fmul, and the compiler folds a
// promote(fp32)->mul->demote(bf16) chain back into an unselectable bf16 fmul,
// even across a volatile round-trip around the multiply or a pure bitcast
// conversion (bisheng re-canonicalizes the bit pattern back into an fpext).
// Route the converted bits through a volatile round-trip so the fp32 value is
// opaque to the fpext-fmul sinking combine, and keep the multiply in fp32.
__aicore__ inline float MulNoNanBf16ToFloat(bfloat16_t v)
{
    volatile uint32_t bits = static_cast<uint32_t>(__builtin_bit_cast(uint16_t, v)) << 16;
    return __builtin_bit_cast(float, bits);
}

__aicore__ inline bfloat16_t MulNoNanFloatToBf16(float v)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, v);
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t exponent = bits & 0x7F800000u;
    const uint32_t mantissa = bits & 0x007FFFFFu;
    if (exponent == 0x7F800000u) {
        const uint16_t signExp = static_cast<uint16_t>(sign >> 16) | 0x7F80u;
        if (mantissa == 0) {
            return __builtin_bit_cast(bfloat16_t, signExp); // Inf
        }
        uint16_t bf16Mant = static_cast<uint16_t>(mantissa >> 16);
        if (bf16Mant == 0) {
            bf16Mant = 1u; // ensure NaN stays NaN
        }
        return __builtin_bit_cast(bfloat16_t, static_cast<uint16_t>(signExp | bf16Mant));
    }
    const uint32_t roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
    const uint16_t bf16Bits = static_cast<uint16_t>((bits + roundingBias) >> 16);
    return __builtin_bit_cast(bfloat16_t, bf16Bits);
}

// y = (x2 == 0) ? 0 : x1 * x2  (scalar fallback path)
template <typename T>
__aicore__ inline T MulNoNanScalar(T a, T b)
{
    if constexpr (std::is_same_v<T, float>) {
        return (b == 0.0f) ? 0.0f : a * b;
    } else if constexpr (std::is_same_v<T, half>) {
        float af = static_cast<float>(a);
        float bf = static_cast<float>(b);
        return static_cast<half>((bf == 0.0f) ? 0.0f : af * bf);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        const float af = MulNoNanBf16ToFloat(a);
        const float bf = MulNoNanBf16ToFloat(b);
        return MulNoNanFloatToBf16((bf == 0.0f) ? 0.0f : af * bf);
    } else {
        return (b == static_cast<T>(0)) ? static_cast<T>(0) : static_cast<T>(a * b);
    }
}

template <typename T>
class MulNoNanKernel {
public:
    __aicore__ inline MulNoNanKernel() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const MulNoNanTilingData* tilingData)
    {
        tiling_ = *tilingData;
        x1Addr_ = reinterpret_cast<__gm__ T*>(x1);
        x2Addr_ = reinterpret_cast<__gm__ T*>(x2);
        yAddr_ = reinterpret_cast<__gm__ T*>(y);
        x1Gm_.SetGlobalBuffer(x1Addr_);
        x2Gm_.SetGlobalBuffer(x2Addr_);
        yGm_.SetGlobalBuffer(yAddr_, static_cast<uint64_t>(tiling_.totalNum));

        constexpr int32_t floatBytes = static_cast<int32_t>(sizeof(float));
        pipe_.InitBuffer(x1Que_, 2, MUL_NO_NAN_TILE_ELEMS * static_cast<int32_t>(sizeof(T)));
        pipe_.InitBuffer(x2Que_, 2, MUL_NO_NAN_TILE_ELEMS * static_cast<int32_t>(sizeof(T)));
        pipe_.InitBuffer(yQue_, 2, MUL_NO_NAN_TILE_ELEMS * static_cast<int32_t>(sizeof(T)));
        pipe_.InitBuffer(maskBuf_, MUL_NO_NAN_TILE_ELEMS * static_cast<int32_t>(sizeof(uint8_t)));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(x1fBuf_, MUL_NO_NAN_TILE_ELEMS * floatBytes);
            pipe_.InitBuffer(x2fBuf_, MUL_NO_NAN_TILE_ELEMS * floatBytes);
            pipe_.InitBuffer(yfBuf_, MUL_NO_NAN_TILE_ELEMS * floatBytes);
        }

        maskL_ = maskBuf_.template Get<uint8_t>();
        if constexpr (!std::is_same_v<T, float>) {
            x1fL_ = x1fBuf_.template Get<float>();
            x2fL_ = x2fBuf_.template Get<float>();
            yfL_ = yfBuf_.template Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        constexpr int64_t elemSize = static_cast<int64_t>(sizeof(T));
        constexpr int64_t alignElems = 32 / elemSize; // 8 for fp32, 16 for fp16/bf16
        int64_t total = tiling_.totalNum;
        if (total <= 0) {
            return;
        }

        int64_t begin = 0;
        int64_t end = 0;
        MulNoNanCalcBlockRange(total, alignElems, begin, end);
        if (begin >= end) {
            return;
        }

        int64_t pos = begin;
        while (pos < end) {
            int64_t run1 = MulNoNanInputContigRun(tiling_, tiling_.x1Strides, pos);
            int64_t run2 = MulNoNanInputContigRun(tiling_, tiling_.x2Strides, pos);
            int64_t cnt = (run1 < run2) ? run1 : run2;
            int64_t rem = end - pos;
            if (cnt > rem) {
                cnt = rem;
            }

            int64_t x1Off = MulNoNanInputOffset(tiling_, tiling_.x1Strides, pos);
            int64_t x2Off = MulNoNanInputOffset(tiling_, tiling_.x2Strides, pos);

            // Vector path: large, aligned, 32B-multiple tile.
            bool vecOk = (cnt >= MUL_NO_NAN_VEC_MIN_ELEMS) && MulNoNanAligned32(x1Off, elemSize, cnt) &&
                         MulNoNanAligned32(x2Off, elemSize, cnt) && MulNoNanAligned32(pos, elemSize, cnt);
            if (vecOk) {
                int64_t vecCnt = cnt;
                vecCnt = (vecCnt / MUL_NO_NAN_VEC_REPEAT) * MUL_NO_NAN_VEC_REPEAT;
                if (vecCnt >= MUL_NO_NAN_VEC_MIN_ELEMS) {
                    ProcessVecSegment(pos, vecCnt, x1Off, x2Off);
                    pos += vecCnt;
                    continue;
                }
            }

            // Scalar fallback, bounded to reach 32B alignment of pos so the
            // next iteration can go vector (offsets are pos-aligned for the
            // contiguous case; broadcast segments are naturally small).
            int64_t scalarCnt = cnt;
            int64_t toAlign = alignElems - (pos % alignElems);
            if (toAlign == alignElems) {
                toAlign = 0;
            }
            if (toAlign > 0 && scalarCnt > toAlign) {
                scalarCnt = toAlign;
            }
            for (int64_t i = 0; i < scalarCnt; ++i) {
                T lhs = x1Gm_.GetValue(static_cast<uint64_t>(x1Off + i));
                T rhs = x2Gm_.GetValue(static_cast<uint64_t>(x2Off + i));
                yGm_.SetValue(static_cast<uint64_t>(pos + i), MulNoNanScalar(lhs, rhs));
            }
            PipeBarrier<PIPE_ALL>();
            pos += scalarCnt;
        }
    }

private:
    // Double-buffered 3-stage pipeline over one contiguous, aligned segment.
    __aicore__ inline void ProcessVecSegment(int64_t pos, int64_t cnt, int64_t x1Off, int64_t x2Off)
    {
        int64_t tiles = MulNoNanCeilDiv(cnt, MUL_NO_NAN_TILE_ELEMS);
        CopyInTile(pos, x1Off, x2Off, cnt, 0);
        for (int64_t t = 0; t < tiles; ++t) {
            if (t + 1 < tiles) {
                CopyInTile(pos, x1Off, x2Off, cnt, t + 1);
            }
            ComputeTile(cnt, t);
            CopyOutTile(pos, cnt, t);
        }
    }

    __aicore__ inline void CopyInTile(int64_t pos, int64_t x1Off, int64_t x2Off, int64_t cnt, int64_t tile)
    {
        int64_t tileStart = tile * MUL_NO_NAN_TILE_ELEMS;
        int64_t tileCnt = cnt - tileStart;
        if (tileCnt > MUL_NO_NAN_TILE_ELEMS) {
            tileCnt = MUL_NO_NAN_TILE_ELEMS;
        }
        LocalTensor<T> x1L = x1Que_.AllocTensor<T>();
        GlobalTensor<T> src1;
        src1.SetGlobalBuffer(x1Addr_ + x1Off + tileStart, static_cast<uint32_t>(tileCnt));
        DataCopy(x1L, src1, static_cast<uint32_t>(tileCnt));
        x1Que_.EnQue(x1L);

        LocalTensor<T> x2L = x2Que_.AllocTensor<T>();
        GlobalTensor<T> src2;
        src2.SetGlobalBuffer(x2Addr_ + x2Off + tileStart, static_cast<uint32_t>(tileCnt));
        DataCopy(x2L, src2, static_cast<uint32_t>(tileCnt));
        x2Que_.EnQue(x2L);
    }

    __aicore__ inline void ComputeTile(int64_t cnt, int64_t tile)
    {
        int64_t tileStart = tile * MUL_NO_NAN_TILE_ELEMS;
        int64_t tileCnt = cnt - tileStart;
        if (tileCnt > MUL_NO_NAN_TILE_ELEMS) {
            tileCnt = MUL_NO_NAN_TILE_ELEMS;
        }
        LocalTensor<T> x1L = x1Que_.DeQue<T>();
        LocalTensor<T> x2L = x2Que_.DeQue<T>();
        LocalTensor<T> yL = yQue_.AllocTensor<T>();
        ComputeVec(x1L, x2L, yL, static_cast<uint32_t>(tileCnt));
        yQue_.EnQue(yL);
        x1Que_.FreeTensor(x1L);
        x2Que_.FreeTensor(x2L);
    }

    __aicore__ inline void CopyOutTile(int64_t pos, int64_t cnt, int64_t tile)
    {
        int64_t tileStart = tile * MUL_NO_NAN_TILE_ELEMS;
        int64_t tileCnt = cnt - tileStart;
        if (tileCnt > MUL_NO_NAN_TILE_ELEMS) {
            tileCnt = MUL_NO_NAN_TILE_ELEMS;
        }
        LocalTensor<T> yL = yQue_.DeQue<T>();
        GlobalTensor<T> dst;
        dst.SetGlobalBuffer(yAddr_ + pos + tileStart, static_cast<uint32_t>(tileCnt));
        DataCopy(dst, yL, static_cast<uint32_t>(tileCnt));
        yQue_.FreeTensor(yL);
    }

    // Vectorized compare+mul+select on UB, same DAG semantics as A5 arch35.
    // VSEL_TENSOR_SCALAR_MODE is used (not TENSOR_TENSOR) because the
    // tensor-tensor bitmap mask chain is unreliable on 910B AIV.
    __aicore__ inline void ComputeVec(const LocalTensor<T>& x1L, const LocalTensor<T>& x2L, const LocalTensor<T>& yL,
                                      uint32_t cnt)
    {
        if constexpr (std::is_same_v<T, float>) {
            Mul(yL, x1L, x2L, cnt);
            Compares(maskL_, x2L, static_cast<float>(0), CMPMODE::NE, cnt);
            Select(yL, maskL_, yL, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, cnt);
        } else {
            Cast(x1fL_, x1L, RoundMode::CAST_NONE, cnt);
            Cast(x2fL_, x2L, RoundMode::CAST_NONE, cnt);
            Mul(yfL_, x1fL_, x2fL_, cnt);
            Compares(maskL_, x2fL_, static_cast<float>(0), CMPMODE::NE, cnt);
            Select(yfL_, maskL_, yfL_, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, cnt);
            Cast(yL, yfL_, RoundMode::CAST_RINT, cnt);
        }
    }

    MulNoNanTilingData tiling_;
    __gm__ T* x1Addr_ = nullptr;
    __gm__ T* x2Addr_ = nullptr;
    __gm__ T* yAddr_ = nullptr;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> x1Que_;
    TQue<QuePosition::VECIN, 2> x2Que_;
    TQue<QuePosition::VECOUT, 2> yQue_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> x1fBuf_;
    TBuf<TPosition::VECCALC> x2fBuf_;
    TBuf<TPosition::VECCALC> yfBuf_;

    LocalTensor<uint8_t> maskL_;
    LocalTensor<float> x1fL_;
    LocalTensor<float> x2fL_;
    LocalTensor<float> yfL_;
};

} // namespace
} // namespace AscendC

#endif // MUL_NO_NAN_H_
