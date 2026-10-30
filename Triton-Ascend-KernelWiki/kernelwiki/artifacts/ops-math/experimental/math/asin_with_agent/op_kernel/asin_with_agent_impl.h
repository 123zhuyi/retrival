/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asin_with_agent_impl.h
 * \brief AsinWithAgent Kernel 实现（arch32: Ascend910B）
 *
 * 性能优化版本：
 *   - Group A (TilingKey 0/1)：float/half，手动泰勒展开（分段法）
 *     不使用 AscendC::Asin 高阶 API，消除内部 PipeBarrier 停顿
 *     tmpBuffer = 5 * tileLength * sizeof(float)（5 个 float 工作 buffer）
 *     使用 Mul+Adds 代替 Fma（arch32 不支持 adv_api Fma 标量版本）
 *     使用 CompareScalar+Select 代替 Cmps（arch32 不直接支持 Cmps）
 *   - Group B (TilingKey 2)：DOUBLE 在 op_api 层已转为 fp32；走 Group A fp32 路径
 *   - Group C (TilingKey 3-8)：整数/BOOL，Cast→fp32→Asin<float>（保留高阶 API）
 *
 * Group A 泰勒展开算法（分段法，向量化无分支）：
 *   |x| < 0.7071: arcsin(x) ≈ x*(1 + x²*(c3 + x²*(c5 + x²*(c7 + x²*(c9 + x²*c11)))))
 *   |x| >= 0.7071: arcsin(x) = sign(x) * (π/2 - arcsin(sqrt(1-x²)))
 *   混合：CompareScalar(|x| >= threshold) → Select(r_large, r_small)
 *
 * 5-buffer 使用说明（从 tmpBuf 按 tileLength 步长切片）：
 *   f1: 通用工作 buffer（xf 临时、Horner scratch、r_y）
 *   f2: x²（Part1）→ y（Part2）→ y²（Part3）→ |xf|（Part5）
 *   f3: Horner poly（Part1：poly(x), Part3：poly(y) scratch）→ r_small [PERSISTENT]
 *   f4: r_small [PERSISTENT]
 *   f5: Horner scratch（Part1）→ Horner poly(y)（Part3）→ r_large [PERSISTENT] → uint8 mask（Part5）
 */

#ifndef ASIN_WITH_AGENT_IMPL_H
#define ASIN_WITH_AGENT_IMPL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/math/asin.h"
#include "asin_with_agent_tiling_data.h"
#include "asin_with_agent_tiling_key.h"

namespace NsAsinWithAgent {

using namespace AscendC;

// ============================================================================
// Group A 计算函数：float / half（TilingKey 0/1）
// 手动泰勒展开，5 个 float 工作 buffer
// ============================================================================
template <typename T>
__aicore__ inline void ComputeGroupA(
    LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<float>& f1, LocalTensor<float>& f2, LocalTensor<float>& f3,
    LocalTensor<float>& f4, LocalTensor<float>& f5, uint32_t count)
{
    constexpr float THRESHOLD = 0.7071067811865476f;
    constexpr float PI_OVER_2 = 1.5707963267948966f;
    constexpr float c17 = 2027025.0f / 185794560.0f; // = 0.010910928
    constexpr float c15 = 135135.0f / 9676800.0f;    // = 0.013969844
    constexpr float c13 = 10395.0f / 599040.0f;      // = 0.017352916
    constexpr float c11 = 945.0f / 42240.0f;         // = 0.022372159
    constexpr float c9 = 105.0f / 3456.0f;           // = 0.030381944
    constexpr float c7 = 15.0f / 336.0f;             // = 0.044642857
    constexpr float c5 = 3.0f / 40.0f;               // = 0.075
    constexpr float c3 = 1.0f / 6.0f;                // = 0.166666667

    // ---- 获取 xf 到 f1 ----
    if constexpr (std::is_same<T, half>::value) {
        Cast(f1, src, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(f1, src, 0.0f, count);
    }

    // ---- Part 1: r_small = xf * poly(x²) ----
    // f2 = x²
    Mul(f2, f1, f1, count);
    // Horner: poly in f3, scratch in f5 (for Mul result)
    // 9-term polynomial (up to x^17) for accuracy < 1e-4 at x=0.7071
    Muls(f3, f2, 0.0f, count);
    Adds(f3, f3, c17, count); // f3 = c17
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c15, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c13, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c11, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c9, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c7, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c5, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, c3, count);
    Mul(f5, f2, f3, count);
    Adds(f3, f5, 1.0f, count); // f3 = 1 + x²*...
    Mul(f4, f1, f3, count);    // f4 = xf * poly(x) = r_small

    // ---- Part 2: y = sqrt(1 - x²) ----
    // Overwrite f2 (x² no longer needed)
    // Note: NO Maxs clamp here → for |x|>1.0: 1-x²<0 → Sqrt gives NaN (correct behavior)
    Muls(f2, f2, -1.0f, count);
    Adds(f2, f2, 1.0f, count);
    Sqrt(f2, f2, count); // f2 = y [or NaN for |x|>1]
    PipeBarrier<PIPE_V>();

    // ---- Part 3: r_y = arcsin_taylor(y) ----
    // Save y from f2 to f3 (overwrite stale poly(x), keeping r_small safe in f4)
    Adds(f3, f2, 0.0f, count); // f3 = y
    // f2 = y²
    Mul(f2, f3, f3, count); // f2 = y²  (dst=f2 ≠ src=f3: safe)
    // Horner for arcsin(y): poly in f5, scratch in f1 (overwrite xf; reload later)
    // 9-term polynomial for accuracy near threshold
    Muls(f5, f2, 0.0f, count);
    Adds(f5, f5, c17, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c15, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c13, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c11, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c9, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c7, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c5, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, c3, count);
    Mul(f1, f2, f5, count);
    Adds(f5, f1, 1.0f, count); // f5 = 1 + y²*...
    Mul(f1, f3, f5, count);    // f1 = y * poly(y) = r_y (dst=f1 ≠ src0=f3, src1=f5)

    // ---- Part 4: r_large = sign(xf) * (π/2 - r_y) ----
    // f5 = π/2 - r_y
    Muls(f5, f1, -1.0f, count);
    Adds(f5, f5, PI_OVER_2, count);
    // Reload xf into f3 (overwrite y; y and y² no longer needed)
    if constexpr (std::is_same<T, half>::value) {
        Cast(f3, src, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(f3, src, 0.0f, count);
    }
    // f2 = |xf|
    Abs(f2, f3, count);
    // f1 = (π/2 - r_y) * xf  (dst=f1 ≠ src0=f5, src1=f3: no aliasing)
    Mul(f1, f5, f3, count);
    Maxs(f2, f2, 1e-6f, count); // avoid div by zero
    Div(f5, f1, f2, count);     // f5 = sign(xf) * (π/2 - r_y) = r_large
    PipeBarrier<PIPE_V>();

    // ---- Part 5: blend r_small(f4) and r_large(f5) using float mask ----
    // mask_float = 0.0 if |xf| < threshold, 1.0 if |xf| >= threshold
    // Computed via steep step: max(min((|xf|-threshold)*1e30, 1.0), 0.0)
    Abs(f2, f3, count);              // f2 = |xf|
    Adds(f2, f2, -THRESHOLD, count); // f2 = |xf| - threshold
    Muls(f2, f2, 1.0e30f, count);    // steep multiply
    Maxs(f2, f2, 0.0f, count);       // clamp to [0, +inf)
    Mins(f2, f2, 1.0f, count);       // clamp to [0, 1] → 0 for small x, 1 for large x
    // result = r_small + mask*(r_large - r_small)
    Sub(f1, f5, f4, count); // f1 = r_large - r_small
    Mul(f3, f2, f1, count); // f3 = mask * (r_large - r_small)
    Add(f2, f4, f3, count); // f2 = r_small + mask*(r_large-r_small) = result

    // ---- Part 6: write back dst ----
    if constexpr (std::is_same<T, float>::value) {
        Adds(dst, f2, 0.0f, count);
    } else {
        Cast(dst, f2, RoundMode::CAST_ROUND, count);
        PipeBarrier<PIPE_V>();
    }
}

// ============================================================================
// Group B（TilingKey=2）：DOUBLE 路径 → 与 Group A fp32 相同
// ============================================================================
__aicore__ inline void ComputeGroupB(
    LocalTensor<float>& dst, LocalTensor<float>& src, LocalTensor<float>& f1, LocalTensor<float>& f2,
    LocalTensor<float>& f3, LocalTensor<float>& f4, LocalTensor<float>& f5, uint32_t count)
{
    ComputeGroupA<float>(dst, src, f1, f2, f3, f4, f5, count);
}

// ============================================================================
// Group C 计算函数（TilingKey 3-8，整数/BOOL）
// ============================================================================

__aicore__ inline void ComputeGroupC_Int8(
    LocalTensor<float>& dst, LocalTensor<int8_t>& src, LocalTensor<half>& halfBuf, LocalTensor<float>& floatCastBuf,
    LocalTensor<uint8_t>& tmpBuf, uint32_t count)
{
    AscendC::Cast(halfBuf, src, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Cast(floatCastBuf, halfBuf, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Asin(dst, floatCastBuf, tmpBuf, count);
}

__aicore__ inline void ComputeGroupC_Int16(
    LocalTensor<float>& dst, LocalTensor<int16_t>& src, LocalTensor<float>& castBuf, LocalTensor<uint8_t>& tmpBuf,
    uint32_t count)
{
    AscendC::Cast(castBuf, src, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Asin(dst, castBuf, tmpBuf, count);
}

__aicore__ inline void ComputeGroupC_Int32(
    LocalTensor<float>& dst, LocalTensor<int32_t>& src, LocalTensor<float>& castBuf, LocalTensor<uint8_t>& tmpBuf,
    uint32_t count)
{
    AscendC::Cast(castBuf, src, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Asin(dst, castBuf, tmpBuf, count);
}

__aicore__ inline void ComputeGroupC_Int64(
    LocalTensor<float>& dst, LocalTensor<int64_t>& src, LocalTensor<int32_t>& i32Buf, LocalTensor<float>& floatCastBuf,
    LocalTensor<uint8_t>& tmpBuf, uint32_t count)
{
    AscendC::Cast(i32Buf, src, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Cast(floatCastBuf, i32Buf, RoundMode::CAST_ROUND, count);
    PipeBarrier<PIPE_V>();
    AscendC::Asin(dst, floatCastBuf, tmpBuf, count);
}

__aicore__ inline void ComputeGroupC_Uint8(
    LocalTensor<float>& dst, LocalTensor<uint8_t>& src, LocalTensor<half>& halfBuf, LocalTensor<float>& floatCastBuf,
    LocalTensor<uint8_t>& tmpBuf, uint32_t count)
{
    AscendC::Cast(halfBuf, src, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Cast(floatCastBuf, halfBuf, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Asin(dst, floatCastBuf, tmpBuf, count);
}

__aicore__ inline void ComputeGroupC_Bool(
    LocalTensor<float>& dst, LocalTensor<bool>& src, LocalTensor<half>& halfBuf, LocalTensor<float>& floatCastBuf,
    LocalTensor<uint8_t>& tmpBuf, uint32_t count)
{
    LocalTensor<uint8_t> srcU8 = src.template ReinterpretCast<uint8_t>();
    AscendC::Cast(halfBuf, srcU8, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Cast(floatCastBuf, halfBuf, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    AscendC::Asin(dst, floatCastBuf, tmpBuf, count);
}

// ============================================================================
// 输出类型 trait
// ============================================================================
template <typename D_T>
struct OutputTypeOf {
    using type = D_T;
};
template <>
struct OutputTypeOf<int8_t> {
    using type = float;
};
template <>
struct OutputTypeOf<int16_t> {
    using type = float;
};
template <>
struct OutputTypeOf<int32_t> {
    using type = float;
};
template <>
struct OutputTypeOf<int64_t> {
    using type = float;
};
template <>
struct OutputTypeOf<uint8_t> {
    using type = float;
};
template <>
struct OutputTypeOf<bool> {
    using type = float;
};

// ============================================================================
// AsinWithAgent Kernel 主类（arch32）
//
// tmpBuf 布局（Group A/B，TK0/1/2）：
//   tmpBufferSize = 5 * tileLength * sizeof(float)
//   [f1][f2][f3][f4][f5]，每块 tileLength * sizeof(float) 字节
//
// tmpBuf 布局（Group C，TK3-8）：
//   tmpBuf 直接传给 AscendC::Asin（大小由 GetAsinMaxMinTmpSize 确定）
// ============================================================================
template <typename D_T>
class AsinWithAgent {
    static constexpr int32_t BUFFER_NUM = 2;
    using O_T = typename OutputTypeOf<D_T>::type;

public:
    __aicore__ inline AsinWithAgent() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AsinWithAgentTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t tileIdx, int64_t currentLen);
    __aicore__ inline void Compute(int64_t currentLen);
    __aicore__ inline void CopyOut(int64_t tileIdx, int64_t currentLen);

    template <typename T = D_T>
    __aicore__ inline void ComputeImpl(LocalTensor<O_T>& dst, LocalTensor<T>& src, uint32_t count);

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    // tmpBuf：Group A = 5 float sub-buffers; Group C = Asin API tmpBuf
    TBuf<QuePosition::VECCALC> tmpBufPing;
    TBuf<QuePosition::VECCALC> tmpBufPong;

    // midBuf：Group C 中间类型转换 buffer
    TBuf<QuePosition::VECCALC> midBufPing;
    TBuf<QuePosition::VECCALC> midBufPong;

    GlobalTensor<D_T> inputGM;
    GlobalTensor<O_T> outputGM;

    int64_t blockOffset_;
    int64_t blockLength_;
    int64_t tileLength_;
    int64_t loopCount_;
    int64_t tailTileLength_;
    uint32_t tmpBufferSize_;
    uint32_t midBufferSize_;
    int64_t tileStep_;
};

template <typename D_T>
__aicore__ inline void AsinWithAgent<D_T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const AsinWithAgentTilingData* tilingData)
{
    uint32_t totalLength = tilingData->totalLength;
    uint32_t usedCoreNum = tilingData->usedCoreNum;
    uint32_t perCoreBase = (usedCoreNum > 0) ? (totalLength / usedCoreNum) : 0;
    uint32_t remainder = (usedCoreNum > 0) ? (totalLength % usedCoreNum) : 0;
    uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());

    if (blockIdx < remainder) {
        blockOffset_ = static_cast<int64_t>(blockIdx * (perCoreBase + 1));
        blockLength_ = static_cast<int64_t>(perCoreBase + 1);
    } else {
        blockOffset_ = static_cast<int64_t>(remainder * (perCoreBase + 1) + (blockIdx - remainder) * perCoreBase);
        blockLength_ = static_cast<int64_t>(perCoreBase);
    }

    tileLength_ = static_cast<int64_t>(tilingData->tileLength);
    tmpBufferSize_ = tilingData->tmpBufferSize;
    midBufferSize_ = tilingData->midBufferSize;

    if (blockLength_ > 0 && tileLength_ > 0) {
        loopCount_ = blockLength_ / tileLength_;
        tailTileLength_ = blockLength_ % tileLength_;
    } else {
        loopCount_ = 0;
        tailTileLength_ = 0;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T*)x + blockOffset_, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ O_T*)y + blockOffset_, blockLength_);

    pipe.InitBuffer(inputQueue, BUFFER_NUM, tileLength_ * sizeof(D_T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, tileLength_ * sizeof(O_T));

    if (tmpBufferSize_ > 0) {
        pipe.InitBuffer(tmpBufPing, tmpBufferSize_);
        pipe.InitBuffer(tmpBufPong, tmpBufferSize_);
    }

    if (midBufferSize_ > 0) {
        pipe.InitBuffer(midBufPing, midBufferSize_);
        pipe.InitBuffer(midBufPong, midBufferSize_);
    }

    tileStep_ = 0;
}

template <typename D_T>
__aicore__ inline void AsinWithAgent<D_T>::CopyIn(int64_t tileIdx, int64_t currentLen)
{
    LocalTensor<D_T> xLocal = inputQueue.template AllocTensor<D_T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentLen * sizeof(D_T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGM[tileIdx * tileLength_], copyParams, {false, 0, 0, 0});
    inputQueue.EnQue(xLocal);
}

template <typename D_T>
template <typename T>
__aicore__ inline void AsinWithAgent<D_T>::ComputeImpl(LocalTensor<O_T>& dst, LocalTensor<T>& src, uint32_t count)
{
    bool ping = (tileStep_ % 2 == 0);

    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
        // Group A / B: slice 5 float sub-buffers from tmpBuf
        LocalTensor<float> base = ping ? tmpBufPing.Get<float>() : tmpBufPong.Get<float>();
        uint32_t stride = static_cast<uint32_t>(tileLength_);
        LocalTensor<float> f1 = base[0U];
        LocalTensor<float> f2 = base[stride];
        LocalTensor<float> f3 = base[stride * 2U];
        LocalTensor<float> f4 = base[stride * 3U];
        LocalTensor<float> f5 = base[stride * 4U];

        if constexpr (std::is_same<T, float>::value) {
            ComputeGroupA<float>(dst, src, f1, f2, f3, f4, f5, count);
        } else {
            ComputeGroupA<half>(dst, src, f1, f2, f3, f4, f5, count);
        }
    } else {
        // Group C: pass tmpBuf to AscendC::Asin
        LocalTensor<uint8_t> tmpBuf = ping ? tmpBufPing.Get<uint8_t>() : tmpBufPong.Get<uint8_t>();

        if constexpr (std::is_same<T, int8_t>::value) {
            LocalTensor<half> halfBuf = ping ? midBufPing.Get<half>() : midBufPong.Get<half>();
            LocalTensor<float> floatCastBuf =
                ping ? midBufPing.Get<half>().ReinterpretCast<float>()[static_cast<uint32_t>(tileLength_) / 2U] :
                       midBufPong.Get<half>().ReinterpretCast<float>()[static_cast<uint32_t>(tileLength_) / 2U];
            ComputeGroupC_Int8(dst, src, halfBuf, floatCastBuf, tmpBuf, count);
        } else if constexpr (std::is_same<T, int16_t>::value) {
            LocalTensor<float> castBuf = ping ? midBufPing.Get<float>() : midBufPong.Get<float>();
            ComputeGroupC_Int16(dst, src, castBuf, tmpBuf, count);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            LocalTensor<float> castBuf = ping ? midBufPing.Get<float>() : midBufPong.Get<float>();
            ComputeGroupC_Int32(dst, src, castBuf, tmpBuf, count);
        } else if constexpr (std::is_same<T, int64_t>::value) {
            LocalTensor<int32_t> i32Buf = ping ? midBufPing.Get<int32_t>() : midBufPong.Get<int32_t>();
            LocalTensor<float> floatCastBuf = ping ? midBufPing.Get<float>() : midBufPong.Get<float>();
            ComputeGroupC_Int64(dst, src, i32Buf, floatCastBuf, tmpBuf, count);
        } else if constexpr (std::is_same<T, uint8_t>::value) {
            LocalTensor<half> halfBuf = ping ? midBufPing.Get<half>() : midBufPong.Get<half>();
            LocalTensor<float> floatCastBuf =
                ping ? midBufPing.Get<half>().ReinterpretCast<float>()[static_cast<uint32_t>(tileLength_) / 2U] :
                       midBufPong.Get<half>().ReinterpretCast<float>()[static_cast<uint32_t>(tileLength_) / 2U];
            ComputeGroupC_Uint8(dst, src, halfBuf, floatCastBuf, tmpBuf, count);
        } else if constexpr (std::is_same<T, bool>::value) {
            LocalTensor<half> halfBuf = ping ? midBufPing.Get<half>() : midBufPong.Get<half>();
            LocalTensor<float> floatCastBuf =
                ping ? midBufPing.Get<half>().ReinterpretCast<float>()[static_cast<uint32_t>(tileLength_) / 2U] :
                       midBufPong.Get<half>().ReinterpretCast<float>()[static_cast<uint32_t>(tileLength_) / 2U];
            ComputeGroupC_Bool(dst, src, halfBuf, floatCastBuf, tmpBuf, count);
        }
    }
}

template <typename D_T>
__aicore__ inline void AsinWithAgent<D_T>::Compute(int64_t currentLen)
{
    LocalTensor<D_T> xLocal = inputQueue.template DeQue<D_T>();
    LocalTensor<O_T> yLocal = outputQueue.template AllocTensor<O_T>();

    ComputeImpl<D_T>(yLocal, xLocal, static_cast<uint32_t>(currentLen));

    outputQueue.template EnQue<O_T>(yLocal);
    inputQueue.FreeTensor(xLocal);
    tileStep_++;
}

template <typename D_T>
__aicore__ inline void AsinWithAgent<D_T>::CopyOut(int64_t tileIdx, int64_t currentLen)
{
    LocalTensor<O_T> yLocal = outputQueue.template DeQue<O_T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint16_t>(currentLen * sizeof(O_T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[tileIdx * tileLength_], yLocal, copyParams);
    outputQueue.FreeTensor(yLocal);
}

template <typename D_T>
__aicore__ inline void AsinWithAgent<D_T>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }

    for (int64_t i = 0; i < loopCount_; i++) {
        CopyIn(i, tileLength_);
        Compute(tileLength_);
        CopyOut(i, tileLength_);
    }

    if (tailTileLength_ > 0) {
        CopyIn(loopCount_, tailTileLength_);
        Compute(tailTileLength_);
        CopyOut(loopCount_, tailTileLength_);
    }
}

} // namespace NsAsinWithAgent

#endif // ASIN_WITH_AGENT_IMPL_H
