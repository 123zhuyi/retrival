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
 * \file cdist_grad_p2.h
 * \brief CdistGrad p=2 (Euclidean) — replicates 950 CdistGradP2Dag exactly.
 *
 *   result = grad * diff / (cdist + 1e-38) * (cdist != 0 ? 1 : 0)
 *   (950: Mul(CastGrad, OpDiff) -> Div(numer, SafeCdist=Eps 1e-38) -> Mul(MaskNEZero))
 *
 * Fully vectorized on the broadcast [B,P,Q,M] layout. Select masks are always written at
 * the aligned BASE of a mask buffer (VSEL requires an aligned mask address).
 */

#ifndef CDIST_GRAD_P2_H
#define CDIST_GRAD_P2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "cdist_grad_common.h"

namespace NsCdistGrad {

using namespace AscendC;

template <typename T>
class CdistGradP2 : public CdistGradBase<T, CdistGradP2<T>> {
public:
    using Base = CdistGradBase<T, CdistGradP2<T>>;
    __aicore__ inline void PrepareChunk(int64_t currentRTile);
    __aicore__ inline void ComputeBatch(int64_t base, int64_t rows);
};

template <typename T>
__aicore__ inline void CdistGradP2<T>::PrepareChunk(int64_t currentRTile)
{
    (void)currentRTile; // masks are computed per batch in ComputeBatch
}

template <typename T>
__aicore__ inline void CdistGradP2<T>::ComputeBatch(int64_t base, int64_t rows)
{
    const int64_t off = base * this->mAligned_;
    // Compare/Select ignore a tail shorter than 256B, so the batch count is rounded up;
    // the extra lanes land in the buffers' CMP_ALIGN slack and no row ever reads them.
    const uint32_t n = this->CmpCount(rows * this->mAligned_);
    LocalTensor<float> term = this->term_[off];
    LocalTensor<uint8_t> maskDistZero = this->maskBuf.template Get<uint8_t>();

    // diff = x1 - x2[j]
    this->SubX1(term, off, rows, n);
    // mask = (cdist == 0) on the RAW queue value (matches 950 MaskNEZero)
    AscendC::Compares(maskDistZero, this->distChunk_[off], 0.0f, AscendC::CMPMODE::EQ, n);
    // safe cdist = cdist + 1e-38 into a SCRATCH buffer (matches 950 Eps).
    LocalTensor<float> distSafe = this->sc1_;
    AscendC::Adds(distSafe, this->distChunk_[off], 1e-38f, n);
    // numer = grad * diff   (950: Mul(CastGrad, OpDiff))
    AscendC::Mul(term, this->gradChunk_[off], term, n);
    LocalTensor<float> rcp = this->sc2_;
    LocalTensor<float> t = this->sc3_;
    AscendC::Reciprocal(rcp, distSafe, n); // r0 = 1/b, b = cdist + 1e-38
    AscendC::Mul(t, distSafe, rcp, n);     // b*r0
    AscendC::Muls(t, t, -1.0f, n);         // -b*r0
    AscendC::Adds(t, t, 2.0f, n);          // 2 - b*r0
    AscendC::Mul(rcp, rcp, t, n);          // r1 = r0*(2 - b*r0)
    AscendC::Mul(t, distSafe, rcp, n);     // b*r1
    AscendC::Muls(t, t, -1.0f, n);         // -b*r1
    AscendC::Adds(t, t, 2.0f, n);          // 2 - b*r1
    AscendC::Mul(rcp, rcp, t, n);          // r2 = r1*(2 - b*r1)
    AscendC::Mul(term, term, rcp, n);      // q = a*r2
    // *(cdist != 0): where cdist==0 take 0
    AscendC::Select(term, maskDistZero, this->zero_, term, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
    AscendC::Compares(maskDistZero, this->distChunk_[off], 3.4028235e38f, AscendC::CMPMODE::GE, n);
    AscendC::Select(term, maskDistZero, this->zero_, term, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
}

} // namespace NsCdistGrad

#endif // CDIST_GRAD_P2_H
