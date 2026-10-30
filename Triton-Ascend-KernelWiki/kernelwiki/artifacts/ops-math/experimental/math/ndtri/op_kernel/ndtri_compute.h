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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/*!
 * \file ndtri_compute.h
 * \brief Cephes Ndtri 分区间有理逼近的 Tensor 化实现（FP32 域）。
 *
 * 模块划分（与详细设计 §4.4 对齐）：
 *   - _polevl(x)           : P(x) Horner 多项式
 *   - _plevl(x)            : Q(x) = 1 + ... 首项为 1 的变体
 *   - polevl_plevl(x)      : P(x) / Q(x) 有理函数
 *   - cal_p0(p)            : 中心区 y = sqrt(2π) * pm * (1 + pm^2 * P0(z)/Q0(z))
 *   - cal_sub(q)           : 尾部 x = sqrt(-2 ln q)，x0 = x - ln(x)/x
 *   - cal_p12(x)           : 尾部修正 1/x * P12(1/x)/Q12(1/x)（按 x<8 / x>=8 掩码合并）
 *   - cal_tail(pSafe)      : 尾部 y_tail = sign * (x0 - cal_p12(x))
 *
 * 所有函数在 FP32 域工作。输入/输出 LocalTensor 均由调用者（Kernel 主体）管理 UB 分配，
 * 本文件仅做计算逻辑组合。
 */

#ifndef NDTRI_COMPUTE_H_
#define NDTRI_COMPUTE_H_

#include "kernel_operator.h"
#include "ndtri_coeffs.h"

namespace NsNdtri {

using namespace AscendC;

// ---------------------------------------------------------------
// Cephes polevl 约定（coefs[0] 为最高次项系数）：
//   P(x) = coefs[0]*x^(n-1) + coefs[1]*x^(n-2) + ... + coefs[n-1]
// Horner: ans = coefs[0]; for i=1..n-1: ans = ans*x + coefs[i]
//
// 设计说明（对应 3.2 性能报告"后续优化项"之一）：
//   Horner 每步的 `Mul + Adds` 串联在 arch35 的编译器层已被自动 fuse 为
//   单条 FusedMulAdd 指令；曾于性能优化迭代中手工改写为
//   `Duplicate(scratch, coef) + FusedMulAdd(dst, x, scratch)` 的等价形式，
//   9 组 msprof sweep 对比结果显示大 shape 耗时差 <1.5%，在测量噪声带内，
//   无显著收益；因而保留语义最清晰的原始 Mul+Adds 写法，交由编译器自动融合。
//   详见 docs/performance-report.md §4A（性能优化迭代）。
// ---------------------------------------------------------------
__aicore__ inline void PolEvl(
    const LocalTensor<float>& dst, const LocalTensor<float>& x, const float* coefs, int n,
    const LocalTensor<float>& scratch, int32_t len)
{
    // dst = coefs[0]（最高次）
    Duplicate(dst, coefs[0], len);
    for (int i = 1; i < n; ++i) {
        // scratch = dst * x
        Mul(scratch, dst, x, len);
        // dst = scratch + coefs[i]
        Adds(dst, scratch, coefs[i], len);
    }
}

// ---------------------------------------------------------------
// Cephes p1evl 约定（首项系数为 1，不显式存入 coefs）：
//   Q(x) = x^n + coefs[0]*x^(n-1) + coefs[1]*x^(n-2) + ... + coefs[n-1]
// 等价 Horner: ans = 1; for i=0..n-1: ans = ans*x + coefs[i]
// （FMA 自动融合同 PolEvl。）
// ---------------------------------------------------------------
__aicore__ inline void PlEvl(
    const LocalTensor<float>& dst, const LocalTensor<float>& x, const float* coefs, int n,
    const LocalTensor<float>& scratch, int32_t len)
{
    // dst = 1.0（隐式 x^n 项系数）
    Duplicate(dst, 1.0f, len);
    for (int i = 0; i < n; ++i) {
        Mul(scratch, dst, x, len);
        Adds(dst, scratch, coefs[i], len);
    }
}

// ---------------------------------------------------------------
// 有理函数 R(x) = P(x) / Q(x)（P 无首项约束，Q 首项为 1）
//   - tmpP / tmpQ: 存 P(x) / Q(x) 中间结果
//   - scratch    : Horner 内部 scratch
// ---------------------------------------------------------------
__aicore__ inline void PolEvlPlEvl(
    const LocalTensor<float>& dst, const LocalTensor<float>& x, const float* coefsP, int nP, const float* coefsQ,
    int nQ, const LocalTensor<float>& tmpP, const LocalTensor<float>& tmpQ, const LocalTensor<float>& scratch,
    int32_t len)
{
    PolEvl(tmpP, x, coefsP, nP, scratch, len);
    PlEvl(tmpQ, x, coefsQ, nQ, scratch, len);
    Div(dst, tmpP, tmpQ, len);
}

// ---------------------------------------------------------------
// cal_p0: 中心区
//   y = sqrt(2π) * (pm + pm^3 * R(z))
//     = sqrt(2π) * pm * (1 + z * R(z))
//   其中 pm = p - 0.5, z = pm^2
//
// Buffer 约定（由调用者传入，大小 = len * sizeof(float)）：
//   - y       : 输出
//   - p       : 输入
//   - tmpPm   : pm 中间（可以复用 y 做输入→输出 inplace，不推荐；保持独立更清晰）
//   - tmpZ    : z 中间
//   - tmpP    : P0(z) 结果
//   - tmpQ    : Q0(z) 结果
//   - scratch : Horner scratch
// ---------------------------------------------------------------
__aicore__ inline void CalP0(
    const LocalTensor<float>& y, const LocalTensor<float>& p, const LocalTensor<float>& tmpPm,
    const LocalTensor<float>& tmpZ, const LocalTensor<float>& tmpP, const LocalTensor<float>& tmpQ,
    const LocalTensor<float>& scratch, int32_t len)
{
    // pm = p - 0.5
    Adds(tmpPm, p, -0.5f, len);

    // z = pm * pm
    Mul(tmpZ, tmpPm, tmpPm, len);

    // R = P0(z) / Q0(z)
    PolEvlPlEvl(y, tmpZ, LIST_P0, 5, LIST_Q0, 8, tmpP, tmpQ, scratch, len);
    // y 临时存 R(z)

    // y = z * R
    Mul(y, y, tmpZ, len);
    // y = 1 + z * R
    Adds(y, y, 1.0f, len);
    // y = pm * (1 + z * R)
    Mul(y, y, tmpPm, len);
    // y = sqrt(2π) * y
    Muls(y, y, NDTRI_SQRT_2PI, len);
}

// ---------------------------------------------------------------
// cal_sub: 尾部基础
//   x  = sqrt(-2 ln q)
//   x0 = x - ln(x) / x
//
// 输入 q ∈ (0, e^-2]（由调用者在 cal_tail 中通过 q = select(mask_neg, 1 - pSafe, pSafe) 保证），
// pSafe 钳制已确保 q > 0。
//
// Buffer 约定：
//   - x0   : 输出 x0
//   - xOut : 输出 x（供 cal_p12 使用）
//   - q    : 输入
//   - tmp  : 工作 buffer
// ---------------------------------------------------------------
__aicore__ inline void CalSub(
    const LocalTensor<float>& x0, const LocalTensor<float>& xOut, const LocalTensor<float>& q,
    const LocalTensor<float>& tmp, int32_t len)
{
    // tmp = ln(q)
    Ln(tmp, q, len);
    // tmp = -2 * ln(q)
    Muls(tmp, tmp, -2.0f, len);
    // xOut = sqrt(-2 ln q)
    Sqrt(xOut, tmp, len);
    // tmp = ln(xOut)
    Ln(tmp, xOut, len);
    // tmp = ln(x) / x
    Div(tmp, tmp, xOut, len);
    // x0 = x - ln(x) / x
    Sub(x0, xOut, tmp, len);
}

// ---------------------------------------------------------------
// cal_p12: 尾部修正
//   z = 1 / x
//   r1 = z * P1(z) / Q1(z)  （对 x < 8 使用）
//   r2 = z * P2(z) / Q2(z)  （对 x >= 8 使用）
//   corr = select(x < 8, r1, r2)
//
// Buffer 约定：
//   - corr     : 输出
//   - x        : 输入 x = sqrt(-2 ln q)
//   - tmpZ     : z = 1/x
//   - tmpR1    : r1 = P1(z)/Q1(z)
//   - tmpR2    : r2 = P2(z)/Q2(z)
//   - tmpP     : Horner 多项式 P(x) 结果
//   - tmpQ     : Horner 多项式 Q(x) 结果
//   - scratch  : Horner scratch
//   - maskX    : uint8 mask buffer
//
// ISSUE-001：调用者传入的 len 必须是 64 倍数（FP32 下 256B 对齐），
// 由 Kernel 层的 lenAligned 保证。
// ---------------------------------------------------------------
__aicore__ inline void CalP12(
    const LocalTensor<float>& corr, const LocalTensor<float>& x, const LocalTensor<float>& tmpZ,
    const LocalTensor<float>& tmpR1, const LocalTensor<float>& tmpR2, const LocalTensor<float>& tmpP,
    const LocalTensor<float>& tmpQ, const LocalTensor<float>& scratch, const LocalTensor<uint8_t>& maskX, int32_t len)
{
    // z = 1 / x  =>  tmpZ = 1.0, tmpZ /= x
    Duplicate(tmpZ, 1.0f, len);
    Div(tmpZ, tmpZ, x, len);

    // r1 = P1(z) / Q1(z)
    PolEvlPlEvl(tmpR1, tmpZ, LIST_P1, 9, LIST_Q1, 8, tmpP, tmpQ, scratch, len);

    // r2 = P2(z) / Q2(z)
    PolEvlPlEvl(tmpR2, tmpZ, LIST_P2, 9, LIST_Q2, 8, tmpP, tmpQ, scratch, len);

    // mask: x < 8  -> 选 r1，否则 r2
    CompareScalar(maskX, x, NDTRI_X_BOUNDARY, CMPMODE::LT, len);

    // corr_raw = select(mask, r1, r2)
    Select(corr, maskX, tmpR1, tmpR2, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);

    // corr = z * corr_raw
    Mul(corr, corr, tmpZ, len);
}

// ---------------------------------------------------------------
// cal_tail: 尾部完整流程
//   q    = select(maskNeg, 1 - pSafe, pSafe)
//   x    = sqrt(-2 ln q)
//   x0   = x - ln(x)/x
//   corr = cal_p12(x)
//   base = x0 - corr         （Cephes 源码 x0 -= ...）
//   y_tail = select(maskNeg, +base, -base)
//
// Buffer 约定：
//   - yTail     : 输出
//   - pSafe     : 输入（已经 clamp 到 [FLT_MIN, 1-FLT_MIN]）
//   - maskNeg   : p >= 0.5 的掩码
//   - tmpQ      : q 中间（复用为 "1 - pSafe"）
//   - tmpX      : x
//   - tmpX0     : x0
//   - tmpCorr   : corr
//   - tmp1..5   : 5 个 fp32 scratch buffer（供 cal_sub / cal_p12 使用）
//   - maskX     : uint8 scratch mask
// ---------------------------------------------------------------
__aicore__ inline void CalTail(
    const LocalTensor<float>& yTail, const LocalTensor<float>& pSafe, const LocalTensor<uint8_t>& maskNeg,
    const LocalTensor<float>& tmpQ, const LocalTensor<float>& tmpX, const LocalTensor<float>& tmpX0,
    const LocalTensor<float>& tmpCorr,
    const LocalTensor<float>& tmp1, // cal_sub 的 tmp / cal_p12 的 tmpZ
    const LocalTensor<float>& tmp2, // cal_p12 的 tmpR1
    const LocalTensor<float>& tmp3, // cal_p12 的 tmpR2
    const LocalTensor<float>& tmp4, // cal_p12 的 tmpP
    const LocalTensor<float>& tmp5, // cal_p12 的 tmpQ / cal_p12 的 scratch
    const LocalTensor<uint8_t>& maskX, int32_t len)
{
    // Step 1: q = select(maskNeg, 1 - pSafe, pSafe)
    //   oneMinusP = 1 - pSafe
    Muls(tmpQ, pSafe, -1.0f, len);
    Adds(tmpQ, tmpQ, 1.0f, len);
    // Select: maskNeg=1 -> tmpQ (1-pSafe), maskNeg=0 -> pSafe
    Select(tmpQ, maskNeg, tmpQ, pSafe, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);

    // Step 2: x = sqrt(-2 ln q)，x0 = x - ln(x)/x
    //   CalSub 使用 tmp1 作为工作 buffer
    CalSub(tmpX0, tmpX, tmpQ, tmp1, len);

    // Step 3: corr = cal_p12(x)
    //   CalP12 内部 Horner 需要 tmpP / tmpQ / scratch：复用 tmp4 / tmp5 / tmpQ
    //   注意：tmpQ 在此时已经不再需要（q 在 Step 2 中已经消费）
    CalP12(
        tmpCorr, tmpX,
        /*tmpZ  */ tmp1,
        /*tmpR1 */ tmp2,
        /*tmpR2 */ tmp3,
        /*tmpP  */ tmp4,
        /*tmpQ  */ tmp5,
        /*scratch*/ tmpQ, maskX, len);

    // Step 4: base = x0 - corr
    Sub(tmpX0, tmpX0, tmpCorr, len);

    // Step 5: sign：
    //   - p <  0.5 (maskNeg=0)  -> y_tail = -base
    //   - p >= 0.5 (maskNeg=1)  -> y_tail = +base
    Muls(tmpCorr, tmpX0, -1.0f, len); // -base 存 tmpCorr
    // Select: maskNeg=1 -> +base, maskNeg=0 -> -base
    Select(yTail, maskNeg, tmpX0, tmpCorr, SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
}

} // namespace NsNdtri

#endif // NDTRI_COMPUTE_H_
