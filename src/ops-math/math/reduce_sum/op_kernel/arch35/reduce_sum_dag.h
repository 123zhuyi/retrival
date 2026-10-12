/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_sum_dag.h
 * \brief reduce sum dag
 */

#ifndef REDUCE_SUM_DAG_H
#define REDUCE_SUM_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
namespace ReduceSum
{
using namespace AscendC;
using namespace Ops::Base;
template <typename T, typename PromteT>
struct ReduceSumDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct ReduceSumBoolDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<int8_t>, Placeholder::In0<int8_t>>;
    using Cast0 = Bind<Vec::Cast<half, int8_t, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<float, half, 0>, Cast0>;
    using Cast2 = Bind<Vec::Cast<int64_t, float, 1>, Cast1>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<int64_t>, Cast2>;
    using OpCopyOut = Bind<Vec::CopyOut<int64_t>, Placeholder::Out0<int64_t>, ReduceOp0>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace ReduceSum

#endif