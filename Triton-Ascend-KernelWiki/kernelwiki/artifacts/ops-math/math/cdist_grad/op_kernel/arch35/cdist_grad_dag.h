/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdist_grad_dag.h
 * \brief cdist_grad dag — six DAGs, Compare+Select via custom Reg operators
 *
 * Custom operators (Reg CompareScalar + Select, avoids DAG multi-consumer):
 *   CdistGradSignOp:       sign(x) via GT/LT compare → 1.0 / -1.0 / 0.0
 *   CdistGradMaskEQOp:     a == b ? 1.0 : 0.0 via Sub + EQ compare
 *   CdistGradMaskNEZeroOp: x != 0 ? 1.0 : 0.0 via EQ compare (inverted)
 *
 * CdistGradP0Dag:      p == 0   → output zeros
 * CdistGradP1Dag:      p == 1   → grad * sign(diff)
 * CdistGradP2Dag:      p == 2   → grad * diff / cdist
 * CdistGradDag:        0<p<2    → sign * |diff|^(p-1) * grad / cdist^(p-1) * masks
 * CdistGradLargePDag:  p>2      → sign * |diff|^(p-1) * grad / cdist^(p-1) * mask_cdist
 * CdistGradInfDag:     p==inf   → grad * sign * mask(|diff| == cdist)
 */

#ifndef CDIST_GRAD_DAG_H
#define CDIST_GRAD_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#include "cdist_grad_operator.h"

namespace CdistGrad {
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;

constexpr int32_t NORM_MODE_GENERAL = 0; // 0 < p < 2, p != 1
constexpr int32_t NORM_MODE_INF = 1;     // p == inf
constexpr int32_t NORM_MODE_LARGE_P = 2; // p > 2
constexpr int32_t NORM_MODE_P0 = 3;      // p == 0
constexpr int32_t NORM_MODE_P1 = 4;      // p == 1
constexpr int32_t NORM_MODE_P2 = 5;      // p == 2

// ---------------------------------------------------------------------------
// CdistGradP0Dag — p == 0 → output zeros
// Var<0>: zero_scalar = 0.0
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradP0Dag {
    using OpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastGrad = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInGrad>;

    using OpZero = Bind<Vec::Muls<PromoteT>, CastGrad, Placeholder::Var<PromoteT, 0>>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, OpZero>;
    using CastOut = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradP1Dag — p == 1
// sign = CdistGradSignOp(diff)   via Reg CompareScalar GT/LT + Select
// result = grad * sign
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradP1Dag {
    using OpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastGrad = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInGrad>;

    using OpCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX1>;

    using OpCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX2>;

    using OpDiff = Bind<Vec::Sub<PromoteT>, CastX1, CastX2>;
    using OpSign = Bind<CdistGradSignOp<PromoteT>, OpDiff>;

    using OpRes = Bind<Vec::Mul<PromoteT>, CastGrad, OpSign>;

    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, OpRes>;
    using CastOut = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradP2Dag — p == 2
// mask = CdistGradMaskNEZeroOp(cdist)   via Reg
// result = grad * diff / (cdist + eps) * mask
// Matches PyTorch: dist == 0 ? 0 : grad * diff / dist
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradP2Dag {
    using Eps = MAKE_CONST(PromoteT, 1e-38);

    using OpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastGrad = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInGrad>;

    using OpCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CastX1 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX1>;

    using OpCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using CastX2 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInX2>;

    using OpCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using CastCdist = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, OpCopyInCdist>;

    using OpDiff = Bind<Vec::Sub<PromoteT>, CastX1, CastX2>;
    using SafeCdist = Bind<Vec::Adds<PromoteT>, CastCdist, Eps>;          // cdist + eps
    using OpNumerator = Bind<Vec::Mul<PromoteT>, CastGrad, OpDiff>;       // grad * diff
    using OpDivResult = Bind<Vec::Div<PromoteT>, OpNumerator, SafeCdist>; // grad * diff / (cdist+eps)

    // mask: cdist != 0 ? 1.0 : 0.0 — zeros out when cdist=0
    using OpMask = Bind<CdistGradMaskNEZeroOp<PromoteT>, CastCdist>;
    using OpResult = Bind<Vec::Mul<PromoteT>, OpDivResult, OpMask>;

    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, OpResult>;
    using CastOut = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradDag — 0 < p < 2, p != 1
// SPLIT form (matching PyTorch lttdist_calc):
//   result = sign(diff) * |diff|^(p-1) * grad / cdist^(p-1)
// NO eps — uses SelectZeroOp (per-element conditional) to handle cdist==0 and
// diff==0, equivalent to PyTorch's vectorized ternary (cond ? 0 : formula).
// Var<0>: power = p - 1
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradDag {
    using DagCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using DagCastGrad = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInGrad>;

    using DagCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using DagCastX1 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInX1>;

    using DagCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using DagCastX2 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInX2>;

    using DagCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using DagCastCdist = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, DagCopyInCdist>;

    // diff & abs & sign
    using DagDiff = Bind<Vec::Sub<PromoteT>, DagCastX1, DagCastX2>;
    using DagDiffAbs = Bind<Vec::Abs<PromoteT>, DagDiff>;
    using DagSign = Bind<CdistGradSignOp<PromoteT>, DagDiff>;

    // Power directly on raw values (no eps — Select handles edge cases downstream)
    using DagTmpBufDiff = Bind<Vec::Abs<PromoteT>, DagDiffAbs>;
    using DagPowDiff = Bind<CdistGradPowDftOp<PromoteT>, DagDiffAbs, DagTmpBufDiff, Placeholder::Var<PromoteT, 0>>;
    using DagTmpBufCdist = Bind<Vec::Abs<PromoteT>, DagCastCdist>;
    using DagPowCdist = Bind<CdistGradPowDftOp<PromoteT>, DagCastCdist, DagTmpBufCdist, Placeholder::Var<PromoteT, 0>>;

    // term = sign * |diff|^(p-1) * grad / cdist^(p-1)
    using DagSignPow = Bind<Vec::Mul<PromoteT>, DagSign, DagPowDiff>;
    using DagWithGrad = Bind<Vec::Mul<PromoteT>, DagSignPow, DagCastGrad>;
    using DagPerTerm = Bind<Vec::DivHighPrecision<PromoteT>, DagWithGrad, DagPowCdist>;

    // Select: 0 where cdist==0 (replaces Mul*maskCdist, no eps needed)
    using DagSelCdist = Bind<CdistGradSelectZeroOp<PromoteT>, DagPerTerm, DagCastCdist>;
    // Select: 0 where |diff|==0 (covers torch blendv for diff==0 & p<1)
    using DagResult = Bind<CdistGradSelectZeroOp<PromoteT>, DagSelCdist, DagDiffAbs>;

    using DagReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, DagResult>;
    using DagCastOut = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, DagReduceOp0>;
    using DagCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, DagCastOut>;

    using Outputs = Elems<DagCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradLargePDag — p > 2
// SPLIT form (matching PyTorch pdist_calc):
//   result = sign(diff) * |diff|^(p-1) * grad / cdist^(p-1)
//   = diff * |diff|^(p-2) * grad / cdist^(p-1)  (PyTorch's form)
// NO eps — uses SelectZeroOp (per-element conditional) to handle cdist==0.
// No mask_diff (p>2 => |diff|^(p-2) at diff=0 = 0, naturally zero).
// Var<0>: power = p - 1
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradLargePDag {
    using LpCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using LpCastGrad = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInGrad>;

    using LpCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using LpCastX1 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInX1>;

    using LpCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using LpCastX2 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInX2>;

    using LpCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using LpCastCdist = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, LpCopyInCdist>;

    // diff & abs & sign
    using LpDiff = Bind<Vec::Sub<PromoteT>, LpCastX1, LpCastX2>;
    using LpDiffAbs = Bind<Vec::Abs<PromoteT>, LpDiff>;
    using LpSign = Bind<CdistGradSignOp<PromoteT>, LpDiff>;

    // Power directly on raw values (no eps — Select handles edge cases downstream)
    using LpTmpBufDiff = Bind<Vec::Abs<PromoteT>, LpDiffAbs>;
    using LpPowDiff = Bind<CdistGradPowDftOp<PromoteT>, LpDiffAbs, LpTmpBufDiff, Placeholder::Var<PromoteT, 0>>;
    using LpTmpBufCdist = Bind<Vec::Abs<PromoteT>, LpCastCdist>;
    using LpPowCdist = Bind<CdistGradPowDftOp<PromoteT>, LpCastCdist, LpTmpBufCdist, Placeholder::Var<PromoteT, 0>>;

    // term = sign * |diff|^(p-1) * grad / cdist^(p-1)
    using LpSignPow = Bind<Vec::Mul<PromoteT>, LpSign, LpPowDiff>;
    using LpWithGrad = Bind<Vec::Mul<PromoteT>, LpSignPow, LpCastGrad>;
    using LpPerTerm = Bind<Vec::DivHighPrecision<PromoteT>, LpWithGrad, LpPowCdist>;

    // Select: 0 where cdist==0 (replaces Mul*maskCdist, no eps needed)
    using LpResult = Bind<CdistGradSelectZeroOp<PromoteT>, LpPerTerm, LpCastCdist>;

    using LpReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, LpResult>;
    using LpCastOut = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, LpReduceOp0>;
    using LpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, LpCastOut>;

    using Outputs = Elems<LpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ---------------------------------------------------------------------------
// CdistGradInfDag — p == inf
// sign = CdistGradSignOp(diff)       via Reg CompareScalar GT/LT + Select
// mask = CdistGradMaskEQOp(|diff|, cdist)  via Reg Sub + CompareScalar EQ + Select
// result = grad * sign * mask
// ---------------------------------------------------------------------------
template <typename T, typename PromoteT>
struct CdistGradInfDag {
    using InfCopyInGrad = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using InfCastGrad = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInGrad>;

    using InfCopyInX1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using InfCastX1 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInX1>;

    using InfCopyInX2 = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using InfCastX2 = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInX2>;

    using InfCopyInCdist = Bind<Vec::CopyIn<T>, Placeholder::In3<T>>;
    using InfCastCdist = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InfCopyInCdist>;

    using InfDiff = Bind<Vec::Sub<PromoteT>, InfCastX1, InfCastX2>;
    using InfDiffAbs = Bind<Vec::Abs<PromoteT>, InfDiff>;

    // sign = Reg sign(diff)
    using InfSign = Bind<CdistGradSignOp<PromoteT>, InfDiff>;

    // mask = Reg (|diff| == cdist) ? 1.0 : 0.0
    using InfMask = Bind<CdistGradMaskEQOp<PromoteT>, InfDiffAbs, InfCastCdist>;

    // result = grad * sign * mask
    using InfGradSign = Bind<Vec::Mul<PromoteT>, InfCastGrad, InfSign>;
    using InfResult = Bind<Vec::Mul<PromoteT>, InfGradSign, InfMask>;

    using InfReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, InfResult>;
    using InfCastOut = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, InfReduceOp0>;
    using InfCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, InfCastOut>;

    using Outputs = Elems<InfCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace CdistGrad

#endif // CDIST_GRAD_DAG_H
