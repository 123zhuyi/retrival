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
 * \file cdist_grad_pinf.h
 * \brief CdistGrad p=inf — replicates 950 CdistGradInfDag exactly.
 *
 *   result = grad * sign(diff) * (|diff| == cdist ? 1 : 0)
 *   sign = hard sign(diff) (Compare GT/LT + Select)
 *   mask = (|diff| - cdist) == 0  (Sub + CompareScalar EQ, exact equality)
 */

#ifndef CDIST_GRAD_PINF_H
#define CDIST_GRAD_PINF_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradPInf : public CdistGradBase<T, CdistGradPInf<T>> {
public:
    using Base = CdistGradBase<T, CdistGradPInf<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeBatch(int64_t base, int64_t rows);
};

template <typename T>
__aicore__ inline void CdistGradPInf<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // no chunk-level preprocessing needed
}

// term[j] = grad[j] * sign(diff) * (|diff| == cdist[j]) for `rows` j rows in one pass.
template <typename T>
__aicore__ inline void CdistGradPInf<T>::ComputeBatch(int64_t base, int64_t rows)
{
    const int64_t off = base * this->mAligned_;
    const uint32_t n = this->CmpCount(rows * this->mAligned_);
    LocalTensor<float> term = this->term_[off];
    LocalTensor<float> diff = this->sc1_;
    LocalTensor<float> sign = this->sc2_;
    LocalTensor<uint8_t> mask = this->maskBuf.template Get<uint8_t>();

    // diff = x1 - x2[j]
    this->SubX1(diff, off, rows, n);
    // sign(diff): hard decision (matches 950 CdistGradSignOp)
    AscendC::Compares(mask, diff, 0.0f, AscendC::CMPMODE::GT, n);
    AscendC::Select(sign, mask, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    AscendC::Compares(mask, diff, 0.0f, AscendC::CMPMODE::LT, n);
    AscendC::Select(sign, mask, this->negOne_, sign, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

    // |diff| (overwrite diff)
    AscendC::Abs(diff, diff, n);
    // mask = (|diff| - cdist) == 0  (950 CdistGradMaskEQOp: exact equality)
    AscendC::Sub(diff, diff, this->distChunk_[off], n);
    AscendC::Compares(mask, diff, 0.0f, AscendC::CMPMODE::EQ, n);
    AscendC::Select(diff, mask, this->one_, this->zero_, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

    // result = grad * sign * mask  (950: Mul(Mul(CastGrad, InfSign), InfMask))
    AscendC::Mul(sign, this->gradChunk_[off], sign, n);
    AscendC::Mul(term, sign, diff, n);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_PINF_H
