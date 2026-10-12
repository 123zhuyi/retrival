/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file reduce_log_sum_exp_dag.h
 * \brief reduce_log_sum_exp_dag
 */

#ifndef CANN_CUSTOM_OPS_REDUCE_LOG_SUM_EXP_DAG_H
#define CANN_CUSTOM_OPS_REDUCE_LOG_SUM_EXP_DAG_H

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#include "reduce_log_sum_exp_operator.h"

namespace ReduceLogSumExp {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, typename PromteT = float>
struct ReduceLogSumExpDag {
  using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
  using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;

  using OpExp = Bind<Vec::Exp<PromteT>, OpCopyIn0Cast>;
  using OpReduceSum = Bind<AscendC::ReduceLogSumExpVec::ReduceLogSumExpOp<PromteT>, OpExp>;
  using OpLog = Bind<Vec::Log<PromteT>, OpReduceSum>;
  using OpLogCast = Bind<Vec::Cast<T, PromteT, 1>, OpLog>;

  using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpLogCast>;
  using Outputs = Elems<OpCopyOut>;
  using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
  using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace reducelogsumexp

#endif  // CANN_CUSTOM_OPS_REDUCE_LOG_SUM_EXP_DAG_H