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
 * \file reduce_nansum_dag.h
 * \brief ReduceNansum DAG for ascend950 (arch35).
 *
 * DAG pipeline:
 *   CopyIn(x) -> Cast(T->PromteT) -> ReduceNansumOp -> Cast(PromteT->T) -> CopyOut(y)
 *
 * NaN detection and replacement is integrated into ReduceNansumOp's CopyIn method,
 * which performs NaN->0 replacement (Compare(x,x,EQ) + Select(mask,x,0)) on each
 * data chunk as it is loaded from GM to UB, before the reduction computation.
 *
 * This follows the same architectural pattern as reduce_log_sum_exp:
 * - No intermediate Copy nodes between Cast and the reduce operator
 * - Single consumer per DAG node output, eliminating buffer conflicts
 * - Custom ReduceOp handles preprocessing internally
 *
 * NaN detection: NaN != NaN (IEEE 754), so Compare(x, x, EQ) returns false for NaN -> mask=0 -> Select picks zero.
 * Non-NaN values: Compare(x, x, EQ) returns true -> mask=1 -> Select picks original value.
 */

#ifndef REDUCE_NANSUM_DAG_H
#define REDUCE_NANSUM_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#include "reduce_nansum_operator.h"

namespace ReduceNansum {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, typename PromteT = float>
struct ReduceNansumDag {
    // Input: CopyIn(x) -> Cast to promote type (FP32 for FP16/BF16, same for FP32)
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn>;

    // Custom ReduceOp with integrated NaN->0 preprocessing.
    // The NaN->0 replacement (Compare + Select) is done inside ReduceNansumOp::CopyIn,
    // eliminating the need for separate Compare, Select, and Duplicate DAG nodes.
    // This avoids multi-consumer buffer conflicts in the reduce pipeline.
    using ReduceOp0 = Bind<AscendC::ReduceNansumVec::ReduceNansumOp<PromteT>, Cast0>;

    // Output: Cast back to original type -> CopyOut
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ReduceNansum

#endif  // REDUCE_NANSUM_DAG_H