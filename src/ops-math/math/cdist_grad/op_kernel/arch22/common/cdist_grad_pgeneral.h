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
 * \file cdist_grad_pgeneral.h
 * \brief CdistGrad general p (0 < p < inf, p != 1, p != 2) — replicates 950 CdistGradDag /
 *        CdistGradLargePDag.
 *
 *   result = sign * |diff|^(p-1) * grad / cdist^(p-1)
 *            then SelectZero(cdist==0), SelectZero(|diff|==0)
 *
 * Reformulated for fp32 as:
 *   result = sign * grad * r^q,   q = |p-1|,   r = |diff|/cdist (p>1) or cdist/|diff| (p<1).
 * Collapsing the two `Power` calls into a single power of a bounded ratio is the part that
 * earns its keep: it halves the transcendental work AND removes the cancellation between two
 * separately-rounded powers.
 *
 * Of the hand-rolled high-precision pieces this file used to stack on top of that, only the
 * ones that were measured to pay for themselves remain. Against a float64 reference on 910B:
 *
 *   DivHighPrec (correctly-rounded quotient)  KEPT, unconditionally.
 *       Its value does not show up in a single term -- per-term error is unchanged by it from
 *       M = 8 upward -- but it is decisive once the j reduce cancels. The quotient feeds ln(),
 *       so its relative error becomes an absolute error on ln(r) that q then scales; on a
 *       long, heavily-cancelling reduce that lands straight in the output. Measured on
 *       [1,40,300,37] p=1.5: 2.1e-6 with it, 7.6e-6 without.
 *   PowIntExp (exact integer power)           KEPT. Cheaper AND more accurate than exp(k*ln r)
 *       for integer k -- 1.5x lower per-term error at p = 3.
 *   LnHighPrec / ExpHighPrec (Newton ln, Cody-Waite exp)   DROPPED for M > 1.
 *       Per-term they measured 0.96x, i.e. very slightly WORSE than the platform Log/Exp,
 *       while costing ~1.25x the kernel time: adv_api's fp32 Log/Exp are accurate enough that
 *       two Newton iterations only re-round an already correctly-rounded result.
 *
 * The reason the elementary functions cannot help for M > 1 is that the term is bounded by the
 * fp32 rounding of the INPUT cdist, which reaches the output scaled by |p-1| (measured mean
 * relative error grows linearly in |p-1|: 5.4e-8 at p = 1.5 to 5.2e-7 at p = 7.3, and is the
 * same for every implementation tested, the legacy TBE one included). Internal precision
 * cannot go below a bound set by the operand.
 *
 * M == 1 is where that bound vanishes -- cdist IS |diff| bit for bit, so the true ratio is 1
 * for every element -- and there the refined ln/exp ARE measurably better, so they are kept for
 * that shape alone (dropping them took [1,16,16,1] p=0.5 from 1.75x better than the legacy TBE
 * implementation to 0.84x, i.e. worse).
 *
 * Select masks are always written at the aligned BASE of a mask buffer (VSEL requires an
 * aligned mask address) — per-row Compare, never offset-indexed chunk masks.
 */

#ifndef CDIST_GRAD_PGENERAL_H
#define CDIST_GRAD_PGENERAL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/math/log.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

// Largest finite fp32. Used to detect a cdist that overflowed to +inf, which happens for
// every fp16 case whose true distance exceeds 65504 (the reference computes cdist in fp64
// and rounds it to the operator dtype).
constexpr float MAX_FINITE_F32 = 3.4028235e38f;

// fp32 binary layout, used by the bit tricks below.
constexpr int32_t FP32_MANTISSA_BITS = 23;
constexpr int32_t FP32_EXP_BIAS = 127;
// Dekker split point: clearing the low FP32_SPLIT_LOW_BITS of the 24-bit significand leaves a
// head of <= 12 bits, so head*head and head*tail are representable exactly.
constexpr int32_t FP32_SPLIT_LOW_BITS = 12;
// Exponent range 2^m can hold without producing a denormal or an inf.
constexpr float EXP_SCALE_MIN = -127.0f;
constexpr float EXP_SCALE_MAX = 128.0f;
// Above this exponent the repeated-multiplication chain costs more than exp(k*ln r).
constexpr int64_t POW_INT_EXP_MAX = 16;
// Floor applied to the power base so ln() never sees a zero.
constexpr float POW_BASE_FLOOR = 1e-30f;

// Layout of the p-general scratch pool. `tmp` holds NUM_POW_SCRATCH_ROWS (see the host tiling)
// rows of `count` fp32 elements each; every helper below addresses row i as tmp[i * count].
// DivHighPrec and PowGeneral never hold the pool at the same time, so they reuse the same rows.
constexpr uint32_t DIV_SLOT_RCP = 0;    // 1/den
constexpr uint32_t DIV_SLOT_Q0 = 1;     // first estimate of num/den
constexpr uint32_t DIV_SLOT_PH = 2;     // fl(den*q0), the high half of the product
constexpr uint32_t DIV_SLOT_ACC = 3;    // den*q0 - ph exactly, then the residual
constexpr uint32_t DIV_SLOT_PROD = 4;   // partial products
constexpr uint32_t DIV_SLOT_DEN_LO = 5; // den - denHi
constexpr uint32_t DIV_SLOT_Q0_LO = 6;  // q0 - q0Hi
constexpr uint32_t DIV_SLOT_DEN_HI = 7; // den with its low significand bits cleared
constexpr uint32_t DIV_SLOT_Q0_HI = 8;  // q0 with its low significand bits cleared

constexpr uint32_t POW_SLOT_BASE = 0; // base clamped away from zero
constexpr uint32_t POW_SLOT_LN = 1;   // ln(base), then exp * ln(base)
constexpr uint32_t POW_SLOT_NEG = 2;  // LnHighPrec: -ln estimate
constexpr uint32_t POW_SLOT_AUX = 3;  // LnHighPrec: Newton correction
constexpr uint32_t POW_SLOT_Z = 4;    // ExpHighPrec: x/ln2, then the reduced argument
constexpr uint32_t POW_SLOT_M = 5;    // ExpHighPrec: floor(x/ln2)
constexpr uint32_t POW_SLOT_G = 6;    // ExpHighPrec: c in [0, ln2)
constexpr uint32_t POW_SLOT_I32 = 7;  // ExpHighPrec: 2^m assembled in the exponent field

// dst = src^expInt via repeated multiplication for small integer expInt (exact, no exp/ln).
__aicore__ inline void PowIntExp(LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t expInt, uint32_t count)
{
    AscendC::Adds(dst, src, 0.0f, count); // dst = src
    for (int64_t k = 1; k < expInt; ++k) {
        AscendC::Mul(dst, src, dst, count); // dst = src * dst
    }
}

__aicore__ inline void SplitHead(LocalTensor<int32_t>& hiI, const LocalTensor<float>& src, uint32_t count)
{
    LocalTensor<float> s = src;
    AscendC::ShiftRight<int32_t>(hiI, s.template ReinterpretCast<int32_t>(), FP32_SPLIT_LOW_BITS, count);
    AscendC::ShiftLeft<int32_t>(hiI, hiI, FP32_SPLIT_LOW_BITS, count);
}

__aicore__ inline void DivHighPrec(LocalTensor<float>& dst, const LocalTensor<float>& num,
                                   const LocalTensor<float>& den, LocalTensor<float>& tmp, uint32_t count)
{
    uint32_t e = count;
    LocalTensor<float> rcp = tmp[DIV_SLOT_RCP * e];
    LocalTensor<float> q0 = tmp[DIV_SLOT_Q0 * e];
    LocalTensor<float> ph = tmp[DIV_SLOT_PH * e];
    LocalTensor<float> acc = tmp[DIV_SLOT_ACC * e];
    LocalTensor<float> prod = tmp[DIV_SLOT_PROD * e];
    LocalTensor<float> denLo = tmp[DIV_SLOT_DEN_LO * e];
    LocalTensor<float> q0Lo = tmp[DIV_SLOT_Q0_LO * e];
    LocalTensor<float> denHi = tmp[DIV_SLOT_DEN_HI * e];
    LocalTensor<float> q0Hi = tmp[DIV_SLOT_Q0_HI * e];
    LocalTensor<int32_t> denHiI = denHi.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> q0HiI = q0Hi.template ReinterpretCast<int32_t>();

    AscendC::Reciprocal(rcp, den, count); // r0 = 1/den
    AscendC::Mul(ph, den, rcp, count);    // den*r0
    AscendC::Muls(ph, ph, -1.0f, count);  // -den*r0
    AscendC::Adds(ph, ph, 2.0f, count);   // 2 - den*r0
    AscendC::Mul(rcp, rcp, ph, count);    // r1 = r0*(2 - den*r0)
    AscendC::Mul(ph, den, rcp, count);    // den*r1
    AscendC::Muls(ph, ph, -1.0f, count);  // -den*r1
    AscendC::Adds(ph, ph, 2.0f, count);   // 2 - den*r1
    AscendC::Mul(rcp, rcp, ph, count);    // r2 = r1*(2 - den*r1)
    AscendC::Mul(q0, num, rcp, count);    // q0 = fl(num * r2)

    // Two-product: den*q0 == ph + acc exactly.
    AscendC::Mul(ph, den, q0, count);
    SplitHead(denHiI, den, count);
    AscendC::Sub(denLo, den, denHi, count);
    SplitHead(q0HiI, q0, count);
    AscendC::Sub(q0Lo, q0, q0Hi, count);
    AscendC::Mul(acc, denHi, q0Hi, count);
    AscendC::Sub(acc, acc, ph, count);
    AscendC::Mul(prod, denHi, q0Lo, count);
    AscendC::Add(acc, acc, prod, count);
    AscendC::Mul(prod, denLo, q0Hi, count);
    AscendC::Add(acc, acc, prod, count);
    AscendC::Mul(prod, denLo, q0Lo, count);
    AscendC::Add(acc, acc, prod, count); // acc = den*q0 - ph

    // residual = num - den*q0 = (num - ph) - acc, exact; q1 = q0 + residual/den.
    AscendC::Sub(prod, num, ph, count);
    AscendC::Sub(prod, prod, acc, count);
    AscendC::Mul(prod, prod, rcp, count);
    AscendC::Add(dst, q0, prod, count);
}

__aicore__ inline void ExpHighPrec(LocalTensor<float>& dst, const LocalTensor<float>& x, LocalTensor<float>& z,
                                   LocalTensor<float>& m, LocalTensor<float>& g, LocalTensor<int32_t>& i32,
                                   uint32_t count)
{
    constexpr float LOG2E = 1.4426950408889634f;      // 1/ln2
    constexpr float LN2_HI = 0.693145751953125f;      // ln2 head, 14 significant bits
    constexpr float LN2_LO = 1.4286067653301868e-06f; // ln2 - LN2_HI

    AscendC::Muls(z, x, LOG2E, count);                                        // z ~ x/ln2
    AscendC::Cast<float, float>(m, z, AscendC::RoundMode::CAST_FLOOR, count); // m = floor(z)
    AscendC::Muls(g, m, LN2_HI, count);                                       // m*LN2_HI (exact)
    AscendC::Sub(z, x, g, count);                                             // x - m*LN2_HI
    AscendC::Muls(g, m, LN2_LO, count);                                       // m*LN2_LO
    AscendC::Sub(g, z, g, count);                                             // c in [0, ln2)

    // expm1(c) = c*(1/1! + c*(1/2! + c*(... + c/9!)))
    AscendC::Duplicate(dst, 2.7557319e-06f, count); // 1/9!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 2.4801587e-05f, count); // 1/8!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.9841270e-04f, count); // 1/7!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.3888889e-03f, count); // 1/6!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 8.3333333e-03f, count); // 1/5!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 4.1666667e-02f, count); // 1/4!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.6666667e-01f, count); // 1/3!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 5.0e-01f, count); // 1/2!
    AscendC::Mul(dst, dst, g, count);
    AscendC::Adds(dst, dst, 1.0f, count); // 1/1!
    AscendC::Mul(dst, dst, g, count);     // expm1(c)
    AscendC::Adds(dst, dst, 1.0f, count); // e^c

    AscendC::Maxs(m, m, EXP_SCALE_MIN, count);
    AscendC::Mins(m, m, EXP_SCALE_MAX, count);
    AscendC::Cast<int32_t, float>(i32, m, AscendC::RoundMode::CAST_FLOOR, count);
    AscendC::Adds<int32_t>(i32, i32, FP32_EXP_BIAS, count);           // biased exponent
    AscendC::ShiftLeft<int32_t>(i32, i32, FP32_MANTISSA_BITS, count); // into the exponent field
    LocalTensor<float> pw2 = i32.template ReinterpretCast<float>();   // 2^m exact
    AscendC::Mul(dst, dst, pw2, count);                               // e^c * 2^m
}

__aicore__ inline void LnHighPrec(LocalTensor<float>& res, const LocalTensor<float>& x, LocalTensor<float>& neg,
                                  LocalTensor<float>& aux, LocalTensor<float>& z, LocalTensor<float>& m,
                                  LocalTensor<float>& g, LocalTensor<int32_t>& i32, uint32_t count)
{
    AscendC::Log(res, x, count);                // res = ln(x) seed
    AscendC::Muls(neg, res, -1.0f, count);      // neg = -ln0
    ExpHighPrec(aux, neg, z, m, g, i32, count); // e^{-ln0}
    AscendC::Mul(aux, x, aux, count);           // x * e^{-ln0}
    AscendC::Adds(aux, aux, -1.0f, count);      // x*e^{-ln0} - 1
    AscendC::Add(res, res, aux, count);         // ln1
    AscendC::Muls(neg, res, -1.0f, count);      // -ln1
    ExpHighPrec(aux, neg, z, m, g, i32, count); // e^{-ln1}
    AscendC::Mul(aux, x, aux, count);           // x * e^{-ln1}
    AscendC::Adds(aux, aux, -1.0f, count);      // x*e^{-ln1} - 1
    AscendC::Add(res, res, aux, count);         // ln2 (high precision)
}

__aicore__ inline void PowGeneral(LocalTensor<float>& dst, const LocalTensor<float>& base, float exp,
                                  LocalTensor<float>& tmp, bool exact, uint32_t count)
{
    int64_t eInt = static_cast<int64_t>(exp);
    if (exp == static_cast<float>(eInt) && eInt >= 1 && eInt <= POW_INT_EXP_MAX) {
        PowIntExp(dst, base, eInt, count);
        return;
    }
    uint32_t e = count;
    LocalTensor<float> baseBuf = tmp[POW_SLOT_BASE * e];
    LocalTensor<float> lnBuf = tmp[POW_SLOT_LN * e];
    AscendC::Adds(baseBuf, base, POW_BASE_FLOOR, count); // clamp to avoid ln(0)
    if (!exact) {
        AscendC::Log(lnBuf, baseBuf, count);     // ln(base)
        AscendC::Muls(lnBuf, lnBuf, exp, count); // exp * ln(base)
        AscendC::Exp(dst, lnBuf, count);         // base^exp
        return;
    }
    LocalTensor<float> negBuf = tmp[POW_SLOT_NEG * e];
    LocalTensor<float> auxBuf = tmp[POW_SLOT_AUX * e];
    LocalTensor<float> zBuf = tmp[POW_SLOT_Z * e];
    LocalTensor<float> mBuf = tmp[POW_SLOT_M * e];
    LocalTensor<float> gBuf = tmp[POW_SLOT_G * e];
    LocalTensor<int32_t> i32Buf = tmp[POW_SLOT_I32 * e].template ReinterpretCast<int32_t>();
    LnHighPrec(lnBuf, baseBuf, negBuf, auxBuf, zBuf, mBuf, gBuf, i32Buf, count);
    AscendC::Muls(lnBuf, lnBuf, exp, count);
    ExpHighPrec(dst, lnBuf, zBuf, mBuf, gBuf, i32Buf, count);
}

template <typename T>
class CdistGradPGeneral : public CdistGradBase<T, CdistGradPGeneral<T>> {
public:
    using Base = CdistGradBase<T, CdistGradPGeneral<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeBatch(int64_t base, int64_t rows);
    __aicore__ inline void AccumulateBatch(int64_t base, int64_t rows);
    __aicore__ inline void ResetAccumCompensation();
    __aicore__ inline void FoldAccumCompensation();
};

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::ResetAccumCompensation()
{
    if (this->pValueF_ < 1.0f) {
        return;
    }
    AscendC::Duplicate(this->wsReadBuf.template Get<float>(), 0.0f,
                       static_cast<uint32_t>(this->pTile_ * this->mAligned_));
}

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::FoldAccumCompensation()
{
    if (this->pValueF_ < 1.0f) {
        return;
    }
    AscendC::Add(this->accum_, this->accum_, this->wsReadBuf.template Get<float>(),
                 static_cast<uint32_t>(this->pTile_ * this->mAligned_));
}

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // masks are computed per batch in ComputeBatch
}

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::ComputeBatch(int64_t base, int64_t rows)
{
    const int64_t off = base * this->mAligned_;
    const uint32_t count = this->CmpCount(rows * this->mAligned_);
    LocalTensor<float> term = this->term_[off];
    LocalTensor<float> diff = this->sc1_;
    LocalTensor<float> sign = this->sc2_;
    LocalTensor<float> powDst = this->sc3_;
    LocalTensor<uint8_t> maskDiffZero = this->maskBuf2.template Get<uint8_t>();
    LocalTensor<uint8_t> maskDistZero = this->maskBuf.template Get<uint8_t>();
    float pMinus1 = this->pValueF_ - 1.0f;
    float q = pMinus1 >= 0.0f ? pMinus1 : -pMinus1; // |p-1|

    // diff = x1 - x2[j]
    this->SubX1(diff, off, rows, count);
    // sign(diff): hard decision
    AscendC::Compares(maskDiffZero, diff, 0.0f, AscendC::CMPMODE::GT, count);
    AscendC::Select(sign, maskDiffZero, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    AscendC::Compares(maskDiffZero, diff, 0.0f, AscendC::CMPMODE::LT, count);
    AscendC::Select(sign, maskDiffZero, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

    // |diff|, remember |diff|==0
    AscendC::Abs(diff, diff, count);
    AscendC::Compares(maskDiffZero, diff, 0.0f, AscendC::CMPMODE::EQ, count);

    LocalTensor<float> tmpF32 = this->tmpBuf.template Get<float>();

    const bool exactPath = (this->mSize_ == 1);
    if (pMinus1 >= 0.0f) {
        DivHighPrec(diff, diff, this->distChunk_[off], tmpF32, count);
    } else {
        DivHighPrec(diff, this->distChunk_[off], diff, tmpF32, count);
    }
    // powDst = r^q
    PowGeneral(powDst, diff, q, tmpF32, exactPath, count);
    // sign * r^q * grad
    AscendC::Mul(term, sign, powDst, count);
    AscendC::Mul(term, this->gradChunk_[off], term, count);
    // SelectZero(cdist==0)
    AscendC::Compares(maskDistZero, this->distChunk_[off], 0.0f, AscendC::CMPMODE::EQ, count);
    AscendC::Select(term, maskDistZero, this->zero_, term, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    if (pMinus1 >= 0.0f) {
        AscendC::Compares(maskDistZero, this->distChunk_[off], MAX_FINITE_F32, AscendC::CMPMODE::GE, count);
        AscendC::Select(term, maskDistZero, this->zero_, term, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    }
    // SelectZero(|diff|==0)
    AscendC::Select(term, maskDiffZero, this->zero_, term, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
}

template <typename T>
__aicore__ inline void CdistGradPGeneral<T>::AccumulateBatch(int64_t base, int64_t rows)
{
    const uint32_t count = static_cast<uint32_t>(this->mAligned_);
    if (this->pValueF_ < 1.0f) {
        // For p < 1 a plain add is deliberately more accurate than compensating it -- see
        // ResetAccumCompensation. The base class handles both the blocked and the row form.
        Base::AccumulateBatch(base, rows);
        return;
    }
    LocalTensor<float> comp = this->wsReadBuf.template Get<float>();
    LocalTensor<float> s2 = this->sc1_;
    LocalTensor<float> bv = this->sc2_;
    LocalTensor<float> av = this->sc3_;
    if (this->pTile_ > 1) {
        const int64_t pOut = rows / this->rSize_;
        const uint32_t n = static_cast<uint32_t>(pOut * this->mAligned_);
        const uint64_t mask = static_cast<uint64_t>(this->mAligned_);
        const uint8_t reps = static_cast<uint8_t>(pOut);
        const uint8_t rowBlocks = static_cast<uint8_t>(this->mAligned_ / (BLOCK_BYTES / sizeof(float)));
        const uint8_t termRep = static_cast<uint8_t>(this->rSize_ * rowBlocks);
        const AscendC::BinaryRepeatParams src1Strided(1, 1, 1, rowBlocks, rowBlocks, termRep);
        const AscendC::BinaryRepeatParams src0Strided(1, 1, 1, rowBlocks, termRep, rowBlocks);
        for (int64_t j = 0; j < this->rSize_; j++) {
            LocalTensor<float> x = this->term_[(base + j) * this->mAligned_];
            AscendC::Add(s2, this->accum_, x, mask, reps, src1Strided); // s2 = s + x
            AscendC::Sub(bv, s2, this->accum_, n);                      // bv = s2 - s
            AscendC::Sub(av, s2, bv, n);
            AscendC::Sub(av, this->accum_, av, n);            // s - (s2 - bv)
            AscendC::Sub(bv, x, bv, mask, reps, src0Strided); // x - bv
            AscendC::Add(av, av, bv, n);                      // residual
            AscendC::Add(comp, comp, av, n);
            AscendC::Adds(this->accum_, s2, 0.0f, n); // s = s2
        }
        return;
    }
    for (int64_t k = 0; k < rows; k++) {
        LocalTensor<float> x = this->term_[(base + k) * this->mAligned_];
        AscendC::Add(s2, this->accum_, x, count);  // s2 = s + x
        AscendC::Sub(bv, s2, this->accum_, count); // bv = s2 - s
        AscendC::Sub(av, s2, bv, count);           // av = s2 - bv
        AscendC::Sub(av, this->accum_, av, count); // s - av   (part of s lost in s2)
        AscendC::Sub(bv, x, bv, count);            // x - bv   (part of x lost in s2)
        AscendC::Add(av, av, bv, count);           // residual
        AscendC::Add(comp, comp, av, count);
        AscendC::Adds(this->accum_, s2, 0.0f, count); // s = s2
    }
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_PGENERAL_H
