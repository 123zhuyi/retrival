/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * ------------------------------------------------------------------------
 * Cephes Math Library, Stephen L. Moshier, https://netlib.org/cephes/
 * Licensed under BSD-like terms. Redistribution with attribution.
 * Source: cephes/cprob/ndtri.c
 *
 * The rational approximation coefficients below are derived from the
 * Cephes Math Library (double precision). They are converted to single
 * precision (float) for use on NPU arch35 vector pipeline. The loss of
 * precision compared to double (~7-8 significant decimal digits) is well
 * within the FP32 threshold (2^-13 ≈ 1.22e-4).
 * ------------------------------------------------------------------------
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/*!
 * \file ndtri_coeffs.h
 * \brief Cephes Ndtri 分区间有理逼近系数（FP32）。
 */

#ifndef NDTRI_COEFFS_H_
#define NDTRI_COEFFS_H_

#include "kernel_operator.h"

namespace NsNdtri {

// ---------------------------------------------------------------
// 中心区 |p - 0.5| <= 0.5 - exp(-2)，z = (p - 0.5)^2
// Rational approximation: ndtri(p) ≈ sqrt(2π) * ((p-0.5) + (p-0.5)^3 * P0(z)/Q0(z))
// ---------------------------------------------------------------

// LIST_P0: 5 项（Cephes 约定 polevl：coefs[0] 为最高次 x^(n-1) 的系数）
__aicore__ constexpr float LIST_P0[5] = {
    -5.99633501014107895267e1f, 9.80010754185999661536e1f,  -5.66762857469070293439e1f,
    1.39312609387279679503e1f,  -1.23916583867381258016e0f,
};

// LIST_Q0: 8 项（Cephes 约定 p1evl：x^n 项系数恒为 1，不显式存入；
//           coefs[0] 为 x^(n-1) 的系数，依次降至 coefs[n-1] 为常数项）
__aicore__ constexpr float LIST_Q0[8] = {
    1.95448858338141759834e0f, 4.67627912898881538453e0f,  8.63602421390890590575e1f, -2.25462687854119370527e2f,
    2.00260212380060660359e2f, -8.20372256168333339912e1f, 1.59056225126211695515e1f, -1.18331621121330003142e0f,
};

// ---------------------------------------------------------------
// 尾部区 P1/Q1：x = sqrt(-2 ln q) ∈ [2, 8]
// ---------------------------------------------------------------

// LIST_P1: 9 项
__aicore__ constexpr float LIST_P1[9] = {
    4.05544892305962419923e0f,   3.15251094599893866154e1f,   5.71628192246421288162e1f,
    4.40805073893200834700e1f,   1.46849561928858024014e1f,   2.18663306850790267539e0f,
    -1.40256079171354495875e-1f, -3.50424626827848203418e-2f, -8.57456785154685413611e-4f,
};

// LIST_Q1: 8 项（首项系数为 1，不显式存入）
__aicore__ constexpr float LIST_Q1[8] = {
    1.57799883256466749731e1f, 4.53907635128879210584e1f,   4.13172038254672030440e1f,   1.50425385692907503408e1f,
    2.50464946208309415979e0f, -1.42182922854787788574e-1f, -3.80806407691578277194e-2f, -9.33259480895457427372e-4f,
};

// ---------------------------------------------------------------
// 尾部区 P2/Q2：x = sqrt(-2 ln q) > 8（极端小概率，p < exp(-32)）
// ---------------------------------------------------------------

// LIST_P2: 9 项
__aicore__ constexpr float LIST_P2[9] = {
    3.23774891776946035970e0f,  6.91522889068984211695e0f,  3.93881025292474443415e0f,
    1.33303460815807542389e0f,  2.01485389549179081538e-1f, 1.23716634817820021358e-2f,
    3.01581553508235416007e-4f, 2.65806974686737550832e-6f, 6.23974539184983293730e-9f,
};

// LIST_Q2: 8 项（首项系数为 1，不显式存入）
__aicore__ constexpr float LIST_Q2[8] = {
    6.02427039364742014255e0f,  3.67983563856160859403e0f,  1.37702099489081330271e0f,  2.16236993594496635890e-1f,
    1.34204006088543189037e-2f, 3.28014464682127739104e-4f, 2.89247864745380683936e-6f, 6.79019408009981274425e-9f,
};

// 边界常量
__aicore__ constexpr float NDTRI_VAL_SUB = 0.1353352832366127f;      // e^-2
__aicore__ constexpr float NDTRI_RES_EXP = 0.8646647167633873f;      // 1 - e^-2
__aicore__ constexpr float NDTRI_SQRT_2PI = 2.50662827463100050242f; // sqrt(2π)
__aicore__ constexpr float NDTRI_X_BOUNDARY = 8.0f;                  // P1/Q1 vs P2/Q2 分界
__aicore__ constexpr float NDTRI_SAFE_LO = 1.1754944e-38f;           // FLT_MIN，pSafe 下限

} // namespace NsNdtri

#endif // NDTRI_COEFFS_H_
