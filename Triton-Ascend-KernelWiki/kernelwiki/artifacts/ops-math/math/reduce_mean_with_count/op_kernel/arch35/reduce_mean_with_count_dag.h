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
 * \file reduce_mean_with_count_dag.h
 * \brief reduce mean with count dag
 *
 * DAG pipeline (matches TBE: y = ReduceSum(x * count / count_sum, axes)):
 *   CopyIn(x)       -> Cast(T->PromteT)
 *   CopyIn(count)   -> Cast(T->PromteT)
 *   CopyIn(count_sum) -> Cast(T->PromteT)
 *   Mul(x_casted, count_casted) -> Div(result, count_sum_casted)
 *   ReduceSumOp -> Cast(PromteT->T) -> CopyOut(y)
 */

#ifndef REDUCE_MEAN_WITH_COUNT_DAG_H
#define REDUCE_MEAN_WITH_COUNT_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace ReduceMeanWithCount
{
using namespace Ops::Base;
template <typename T, typename PromteT>
struct ReduceMeanWithCountDag {
    // Input 0: x
    using OpCopyInX        = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastX            = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInX>;

    // Input 1: count (same shape as x, element-wise weight)
    using OpCopyInCount    = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CastCount        = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInCount>;

    // Input 2: count_sum (same shape as x, element-wise divisor)
    using OpCopyInCountSum = Bind<Vec::CopyIn<T>, Placeholder::In2<T>>;
    using CastCountSum     = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInCountSum>;

    // Pre-processing: weighted = x * count / count_sum
    using OpMul            = Bind<Vec::Mul<PromteT>, CastX, CastCount>;
    using OpDiv            = Bind<Vec::Div<PromteT>, OpMul, CastCountSum>;

    // Reduce: ReduceSum on weighted data
    using ReduceOp0        = Bind<Vec::ReduceSumOp<PromteT>, OpDiv>;

    // Output
    using Cast1            = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut        = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ReduceMeanWithCount

#endif  // REDUCE_MEAN_WITH_COUNT_DAG_H
