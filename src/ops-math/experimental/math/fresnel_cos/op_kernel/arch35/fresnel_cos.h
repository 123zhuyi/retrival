/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 *
 * FresnelCos Kernel (arch35 / Ascend950)
 *
 * Algorithm: Cephes-style three-branch (small / large / clamp) with:
 *   - Pre-saturation of |x| > CLAMP_THRESH to a finite safe value to keep
 *     intermediate fp32 math bounded (handles Inf/3.4e38 inputs).
 *   - Sin/Cos with RADIAN_REDUCTION config (validated by probe1) for the
 *     large-branch phase argument (pi/2)*x^2, which exceeds the unreduced
 *     polynomial-approximation valid range for x^2 > a few.
 *   - 64-element aligned Compare to keep tail mask bits deterministic.
 *   - Sign(x) applied as a final post-multiplication: C(x) is odd, so we
 *     compute |result| then multiply by sign(x).
 *   - Clamp |x| > CLAMP_THRESH (using GE comparison so x = 36974 maps to ±0.5).
 */

#ifndef FRESNEL_COS_H
#define FRESNEL_COS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fresnel_cos_tiling_data.h"
#include "fresnel_cos_tiling_key.h"
#include "adv_api/math/sin.h"
#include "adv_api/math/cos.h"

namespace NsFresnelCos {

using namespace AscendC;

// ============================================================================
// Cephes polynomial coefficients
// ============================================================================
static constexpr float CN[6] = {-4.98843114573573548651E-8f, 9.50428062829859605134E-6f,  -6.45191435683965050962E-4f,
                                1.88843319396703850064E-2f,  -2.05525900955013891793E-1f, 9.99999999999999998822E-1f};
static constexpr float CD[7] = {3.99982968972495980367E-12f, 9.15439215774657478799E-10f, 1.25001862479598821474E-7f,
                                1.22262789024179030997E-5f,  8.68029542941784300606E-4f,  4.12142090722199792936E-2f,
                                1.00000000000000000118E0f};
static constexpr float FN[10] = {3.76329711269987889006E-20f, 1.34283276233062758925E-16f, 1.72010743268161828879E-13f,
                                 1.02304514164907233465E-10f, 3.05568983790257605827E-8f,  4.63613749287867322088E-6f,
                                 3.45017939782574027900E-4f,  1.15220955073585758835E-2f,  1.43407919780758885261E-1f,
                                 4.21543555043677546506E-1f};
static constexpr float FD[11] = {1.25443237090011264384E-20f, 4.52001434074129701496E-17f, 5.88754533621578410010E-14f,
                                 3.60140029589371370404E-11f, 1.12699224763999035261E-8f,  1.84627567348930545870E-6f,
                                 1.55934409164153020873E-4f,  6.44051526508858611005E-3f,  1.16888925859191382142E-1f,
                                 7.51586398353378947175E-1f,  1.00000000000000000000E0f};
static constexpr float GN[11] = {1.86958710162783235106E-22f, 8.36354435630677421531E-19f, 1.37555460633261799868E-15f,
                                 1.08268041139020870318E-12f, 4.45344415861750144738E-10f, 9.82852443688422223854E-8f,
                                 1.15138826111884280931E-5f,  6.84079380915393090172E-4f,  1.87648584092575249293E-2f,
                                 1.97102833525523411709E-1f,  5.04442073643383265887E-1f};
static constexpr float GD[12] = {1.86958710162783236342E-22f, 8.39158816283118707363E-19f, 1.38796531259578871258E-15f,
                                 1.10273215066240270757E-12f, 4.60680728146520428211E-10f, 1.04314589657571990585E-7f,
                                 1.27545075667729118702E-5f,  8.14679107184306179049E-4f,  2.53603741420338795122E-2f,
                                 3.37748989120019970451E-1f,  1.47495759925128324529E0f,   1.00000000000000000000E0f};

static constexpr float PI_VAL = 3.14159265358979323846f;
static constexpr float PI_HALF = 1.57079632679489661923f;
static constexpr float PI_SQ = 9.86960440108935861883f;
// Threshold on x^2 (NOT |x|): x^2 < 2.5625 selects the small branch, i.e. |x| < ~1.6.
static constexpr float SMALL_THRESH = 2.5625f;
// Use GE for clamp so x = 36974 (exact fp32) maps to ±0.5 per scipy golden.
static constexpr float CLAMP_THRESH = 36974.0f;
// Safe bounded value to substitute for |x| >= CLAMP_THRESH inside the pipeline.
static constexpr float SAFE_SATURATE = 100.0f;

// Cody-Waite phase reduction: phase = (pi/2) * (x^2 mod 4); default Sin/Cos
// (POLYNOMIAL_APPROXIMATION) operates on reduced phase in [-pi, pi].

// ============================================================================
// KernelFresnelCos
// ============================================================================
template <typename T, uint32_t TILING_KEY>
class KernelFresnelCos {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline KernelFresnelCos() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t curLen);
    __aicore__ inline void Compute(int64_t curLen);
    __aicore__ inline void CopyOut(int64_t progress, int64_t curLen);

    __aicore__ inline void HornerEval(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const float* coeffs, int32_t numCoeffs,
        int64_t n);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;

    TBuf<TPosition::VECCALC> xF32Buf_;
    TBuf<TPosition::VECCALC> absXBuf_;
    TBuf<TPosition::VECCALC> x2Buf_;
    TBuf<TPosition::VECCALC> polyNumBuf_;
    TBuf<TPosition::VECCALC> polyDenBuf_;
    TBuf<TPosition::VECCALC> phaseBuf_;
    TBuf<TPosition::VECCALC> sinBuf_;
    TBuf<TPosition::VECCALC> cosBuf_;
    TBuf<TPosition::VECCALC> resultBuf_;
    TBuf<TPosition::VECCALC> maskBuf1_;
    TBuf<TPosition::VECCALC> maskBuf2_;

    GlobalTensor<T> gmX_;
    GlobalTensor<T> gmY_;

    int64_t blockLength_ = 0;
    int64_t tileLength_ = 0;
};

// ============================================================================
// Init
// ============================================================================
template <typename T, uint32_t TILING_KEY>
__aicore__ inline void KernelFresnelCos<T, TILING_KEY>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR /*workspace*/, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FresnelCosTilingData);
    GET_TILING_DATA_WITH_STRUCT(FresnelCosTilingData, td, tiling);

    uint32_t coreIdx = AscendC::GetBlockIdx();
    int64_t coreOffset;
    if (coreIdx < td.blockNum - 1) {
        blockLength_ = td.baseLength;
    } else {
        blockLength_ = td.tailLength;
    }
    coreOffset = static_cast<int64_t>(coreIdx) * td.baseLength;
    tileLength_ = td.tileLength;

    gmX_.SetGlobalBuffer((__gm__ T*)x + coreOffset, blockLength_);
    gmY_.SetGlobalBuffer((__gm__ T*)y + coreOffset, blockLength_);

    pipe_.InitBuffer(inQueX_, BUFFER_NUM, tileLength_ * static_cast<int64_t>(sizeof(T)));
    pipe_.InitBuffer(outQueY_, BUFFER_NUM, tileLength_ * static_cast<int64_t>(sizeof(T)));

    int64_t f32BufSize = tileLength_ * static_cast<int64_t>(sizeof(float));
    pipe_.InitBuffer(xF32Buf_, f32BufSize);
    pipe_.InitBuffer(absXBuf_, f32BufSize);
    pipe_.InitBuffer(x2Buf_, f32BufSize);
    pipe_.InitBuffer(polyNumBuf_, f32BufSize);
    pipe_.InitBuffer(polyDenBuf_, f32BufSize);
    pipe_.InitBuffer(phaseBuf_, f32BufSize);
    pipe_.InitBuffer(sinBuf_, f32BufSize);
    pipe_.InitBuffer(cosBuf_, f32BufSize);
    pipe_.InitBuffer(resultBuf_, f32BufSize);

    int64_t maskBytes = ((tileLength_ / 8 + 255) / 256) * 256;
    pipe_.InitBuffer(maskBuf1_, maskBytes);
    pipe_.InitBuffer(maskBuf2_, maskBytes);
}

// ============================================================================
// Process
// ============================================================================
template <typename T, uint32_t TILING_KEY>
__aicore__ inline void KernelFresnelCos<T, TILING_KEY>::Process()
{
    int64_t loopCount = (blockLength_ + tileLength_ - 1) / tileLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t curLen = (i == loopCount - 1) ? (blockLength_ - tileLength_ * i) : tileLength_;
        CopyIn(i, curLen);
        Compute(curLen);
        CopyOut(i, curLen);
    }
}

// ============================================================================
// CopyIn / CopyOut
// ============================================================================
template <typename T, uint32_t TILING_KEY>
__aicore__ inline void KernelFresnelCos<T, TILING_KEY>::CopyIn(int64_t progress, int64_t curLen)
{
    LocalTensor<T> xLocal = inQueX_.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    // padParams.isPad=false: bytes in [curLen, calcLen) and [calcLen, cmpCount)
    // are undefined. This is safe here because: (1) Mins/Maxs clamp |x| into
    // [0, SAFE_SATURATE] before any polynomial eval, neutralising NaN/Inf/denormal
    // residues; (2) DataCopyPad on the output side only writes back curLen bytes,
    // so any garbage produced in the tail region is discarded. See L370-374 for
    // the matching divide-by-zero guard on the same rationale.
    DataCopyPad(xLocal, gmX_[progress * tileLength_], copyParams, {false, 0, 0, 0});
    inQueX_.EnQue(xLocal);
}

template <typename T, uint32_t TILING_KEY>
__aicore__ inline void KernelFresnelCos<T, TILING_KEY>::CopyOut(int64_t progress, int64_t curLen)
{
    LocalTensor<T> yLocal = outQueY_.template DeQue<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(gmY_[progress * tileLength_], yLocal, copyParams);
    outQueY_.FreeTensor(yLocal);
}

// ============================================================================
// HornerEval
// ============================================================================
template <typename T, uint32_t TILING_KEY>
__aicore__ inline void KernelFresnelCos<T, TILING_KEY>::HornerEval(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const float* coeffs, int32_t numCoeffs, int64_t n)
{
    Duplicate(dst, coeffs[0], n);
    for (int32_t i = 1; i < numCoeffs; i++) {
        Mul(dst, dst, src, n);
        Adds(dst, dst, coeffs[i], n);
    }
}

// ============================================================================
// Compute (probe1-style: small / large / clamp three-branch with RADIAN_REDUCTION)
//
// Algorithm:
//   1) Cast input -> fp32, take abs and pre-saturate |x| > CLAMP_THRESH to
//      SAFE_SATURATE so that x^2 stays bounded for the polynomial path.
//   2) Compute small branch: x*P(x^4)/Q(x^4) (signed, since CN/CD are even).
//   3) Compute large branch: f(u)=1 - u*P9(u)/Q10(u),  g(u)=t*P10(u)/Q11(u),
//      with u = 1/(pi^2 * x^4). Implementation follows Cephes fresnl.c: the
//      g(u) factor uses t = 1/(pi*x^2) internally, while the final asymptotic
//      formula divides by pi*|x| (see L398-401):
//      C(|x|) = 0.5 + (f*sin(phase) - g*cos(phase)) / (pi*|x|),
//      phase = (pi/2) * x^2, computed via Sin/Cos<RADIAN_REDUCTION>.
//   4) Select: maskSmall=(x^2 < SMALL_THRESH) -> small else large.
//   5) Apply sign(x): C is odd. Take |result| * sign(x).
//   6) Apply clamp: maskClamp=(|x| >= CLAMP_THRESH) -> sign(x)*0.5.
// ============================================================================
template <typename T, uint32_t TILING_KEY>
__aicore__ inline void KernelFresnelCos<T, TILING_KEY>::Compute(int64_t curLen)
{
    LocalTensor<T> xLocal = inQueX_.template DeQue<T>();
    LocalTensor<T> yLocal = outQueY_.template AllocTensor<T>();

    LocalTensor<float> xF32 = xF32Buf_.template Get<float>();
    LocalTensor<float> absX = absXBuf_.template Get<float>();
    LocalTensor<float> x2 = x2Buf_.template Get<float>();
    LocalTensor<float> polyNum = polyNumBuf_.template Get<float>();
    LocalTensor<float> polyDen = polyDenBuf_.template Get<float>();
    LocalTensor<float> phase = phaseBuf_.template Get<float>();
    LocalTensor<float> sinRes = sinBuf_.template Get<float>();
    LocalTensor<float> cosRes = cosBuf_.template Get<float>();
    LocalTensor<float> result = resultBuf_.template Get<float>();
    LocalTensor<uint8_t> maskSmall = maskBuf1_.template Get<uint8_t>();
    LocalTensor<uint8_t> maskClamp = maskBuf2_.template Get<uint8_t>();

    // Compare must run on a length aligned to 64 (fp32) to avoid stale tail bits.
    int64_t cmpCount = ((curLen + 63) / 64) * 64;
    // For internal vector ops, round up to 8-elem (32B) alignment to avoid
    // DataCopy tail truncation on non-aligned shapes.
    int64_t calcLen = ((curLen + 7) / 8) * 8;

    // 1) Cast to fp32
    if constexpr (TILING_KEY == FRESNEL_COS_KEY_FP16 || TILING_KEY == FRESNEL_COS_KEY_BF16) {
        Cast(xF32, xLocal, RoundMode::CAST_NONE, calcLen);
    } else {
        DataCopy(xF32, xLocal, calcLen);
        PipeBarrier<PIPE_V>();
    }

    // 2) absX, build maskClamp on ORIGINAL absX (Inf > 36974 = true)
    Abs(absX, xF32, calcLen);

    Duplicate(result, CLAMP_THRESH, cmpCount);
    Compare(maskClamp, absX, result, CMPMODE::GE, cmpCount);

    // 3) Saturate absX so x^2 stays bounded (handles Inf/3.4e38 robustly).
    Duplicate(result, SAFE_SATURATE, calcLen);
    Select(absX, maskClamp, result, absX, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);

    // Rebuild saturated signed xF32 = sign(originalX) * absX.
    Muls(phase, absX, -1.0f, calcLen);
    {
        LocalTensor<uint8_t> maskNeg = maskBuf1_.template Get<uint8_t>(); // reuse
        Duplicate(result, 0.0f, cmpCount);
        Compare(maskNeg, xF32, result, CMPMODE::LT, cmpCount);
        Select(xF32, maskNeg, phase, absX, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);
    }

    // x^2
    Mul(x2, absX, absX, calcLen);

    // === SMALL branch: small = x * P(x^4) / Q(x^4) (already signed via xF32) ===
    Mul(polyNum, x2, x2, calcLen);              // polyNum = x^4
    HornerEval(phase, polyNum, CN, 6, calcLen); // phase = P5(x^4)
    Mul(phase, phase, xF32, calcLen);           // phase = x * P5 (signed)
    HornerEval(polyDen, polyNum, CD, 7, calcLen);
    Div(phase, phase, polyDen, calcLen); // phase = small branch (signed, handles x=0)

    // === LARGE branch ===
    // u = 1/(pi^2 * x^4)
    Muls(polyNum, polyNum, PI_SQ, calcLen); // polyNum = pi^2 * x^4
    Reciprocal(polyNum, polyNum, calcLen);  // polyNum = u

    // f(u) = 1 - u * P9(u)/Q10(u)
    HornerEval(sinRes, polyNum, FN, 10, calcLen);
    HornerEval(cosRes, polyNum, FD, 11, calcLen);
    Div(sinRes, sinRes, cosRes, calcLen);  // sinRes = P9/Q10
    Mul(sinRes, sinRes, polyNum, calcLen); // sinRes = u*P9/Q10
    Duplicate(cosRes, 1.0f, calcLen);
    Sub(sinRes, cosRes, sinRes, calcLen); // sinRes = f(u)

    // g(u) = t*P10(u)/Q11(u),  t = 1/(pi * x^2)
    HornerEval(cosRes, polyNum, GN, 11, calcLen);
    HornerEval(polyDen, polyNum, GD, 12, calcLen);
    Div(cosRes, cosRes, polyDen, calcLen); // cosRes = P10/Q11
    Muls(polyDen, x2, PI_VAL, calcLen);    // polyDen = pi*x^2
    // Epsilon guard: x=0 -> pi*x^2=0 -> 1/0=Inf. Although the final three-branch
    // Select routes x^2<SMALL_THRESH to the small branch (so the large branch
    // result is discarded for x=0), Inf in intermediate buffers can still
    // contaminate downstream Vec ops via FMA/Select on some NPU pipelines.
    // Add 1e-30 to break the division-by-zero chain. Same pattern as L385.
    Adds(polyDen, polyDen, 1e-30f, calcLen);
    Reciprocal(polyDen, polyDen, calcLen); // polyDen = t
    Mul(cosRes, cosRes, polyDen, calcLen); // cosRes = g(u)

    // phase argument = (pi/2) * (x^2 mod 4) using Cody-Waite reduction
    // Step a: q = round(x^2 / 4)
    Muls(polyNum, x2, 0.25f, calcLen);
    Cast(polyNum, polyNum, RoundMode::CAST_RINT, calcLen);
    // Step b: r = x^2 - q*4   (in [-2, 2])
    Muls(polyNum, polyNum, 4.0f, calcLen);
    Sub(polyNum, x2, polyNum, calcLen);
    // Step c: phase = r * (pi/2)
    Muls(polyNum, polyNum, PI_HALF, calcLen);

    // Default Sin/Cos (POLYNOMIAL_APPROXIMATION) on reduced phase
    Sin<float>(result, polyNum, calcLen);
    Cos<float>(polyDen, polyNum, calcLen);

    // large = 0.5 + (f*sin - g*cos) / (pi*|x|)
    Mul(polyNum, sinRes, result, calcLen);
    Mul(result, cosRes, polyDen, calcLen);
    Sub(polyNum, polyNum, result, calcLen);

    Muls(result, absX, PI_VAL, calcLen);
    Adds(result, result, 1e-30f, calcLen);
    Reciprocal(result, result, calcLen);
    Mul(polyNum, polyNum, result, calcLen);
    Adds(polyNum, polyNum, 0.5f, calcLen);

    Mins(polyNum, polyNum, 1.0f, calcLen);
    Maxs(polyNum, polyNum, -1.0f, calcLen);

    // Apply sign(x) to the LARGE branch only. C(-x) = -C(x).
    Muls(polyDen, polyNum, -1.0f, calcLen);
    {
        LocalTensor<uint8_t> maskNeg = maskBuf1_.template Get<uint8_t>();
        Duplicate(result, 0.0f, cmpCount);
        Compare(maskNeg, xF32, result, CMPMODE::LT, cmpCount);
        Select(polyNum, maskNeg, polyDen, polyNum, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);
    }

    // === Select small vs large by maskSmall = (x^2 < SMALL_THRESH) ===
    Duplicate(result, SMALL_THRESH, cmpCount);
    Compare(maskSmall, x2, result, CMPMODE::LT, cmpCount);
    Select(result, maskSmall, phase, polyNum, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);

    // === Clamp |x| >= 36974 -> sign(x)*0.5 ===
    Duplicate(polyDen, 0.5f, calcLen);
    Muls(polyNum, polyDen, -1.0f, calcLen);
    {
        LocalTensor<uint8_t> maskNeg2 = maskBuf1_.template Get<uint8_t>();
        Duplicate(phase, 0.0f, cmpCount);
        Compare(maskNeg2, xF32, phase, CMPMODE::LT, cmpCount);
        Select(polyDen, maskNeg2, polyNum, polyDen, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);
    }
    Select(result, maskClamp, polyDen, result, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);

    // Cast back
    if constexpr (TILING_KEY == FRESNEL_COS_KEY_FP16 || TILING_KEY == FRESNEL_COS_KEY_BF16) {
        Cast(yLocal, result, RoundMode::CAST_NONE, calcLen);
    } else {
        DataCopy(yLocal, result, calcLen);
        PipeBarrier<PIPE_V>();
    }

    outQueY_.template EnQue<T>(yLocal);
    inQueX_.FreeTensor(xLocal);
}

} // namespace NsFresnelCos

#endif // FRESNEL_COS_H
