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
 * \file cdist_grad_p1.h
 * \brief CdistGrad p=1 — replicates 950 CdistGradP1Dag exactly.
 *
 *   result = grad * sign(diff)
 *   sign(x) = 1.0 if x>0, -1.0 if x<0, 0.0 if x==0  (CompareScalar GT/LT + Select, hard decision)
 *
 * Hard sign (NOT the eps-division approximation) to match 950 precision.
 */

#ifndef CDIST_GRAD_P1_H
#define CDIST_GRAD_P1_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradP1 : public CdistGradBase<T, CdistGradP1<T>> {
public:
    using Base = CdistGradBase<T, CdistGradP1<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeBatch(int64_t base, int64_t rows);
};

template <typename T>
__aicore__ inline void CdistGradP1<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // no chunk-level preprocessing needed
}

// term[j] = grad[j] * sign(x1 - x2[j]) for `rows` consecutive j rows in one pass.
template <typename T>
__aicore__ inline void CdistGradP1<T>::ComputeBatch(int64_t base, int64_t rows)
{
    const int64_t off = base * this->mAligned_;
    const uint32_t n = this->CmpCount(rows * this->mAligned_);
    LocalTensor<float> term = this->term_[off];
    LocalTensor<float> sign = this->sc1_;
    LocalTensor<uint8_t> mask = this->maskBuf.template Get<uint8_t>();

    // diff = x1 - x2[j]
    this->SubX1(term, off, rows, n);
    // sign(diff): x>0 -> 1, x<0 -> -1, x==0 -> 0 (hard decision, matches 950 CdistGradSignOp)
    AscendC::Compares(mask, term, 0.0f, AscendC::CMPMODE::GT, n);
    AscendC::Select(sign, mask, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    AscendC::Compares(mask, term, 0.0f, AscendC::CMPMODE::LT, n);
    AscendC::Select(sign, mask, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    // result = grad * sign   (950: Mul(CastGrad, OpSign))
    AscendC::Mul(term, this->gradChunk_[off], sign, n);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_P1_H
