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
 * \file kl_div_v2_dag.h
 * \brief kl_div_v2 dag
 */

#ifndef KL_DIV_V2_DAG_H
#define KL_DIV_V2_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace KLDivV2 {
using namespace AscendC;
using namespace Ops::Base;

constexpr int CAST_MODE_RINT = 1;
constexpr int COMPARE_MODE_NE = 5;
constexpr int SELECT_MODE_T_S = 1;

// 'sum'和'none',log_target为FALSE的DAG图描述
template <typename T, typename PromteT>
struct KLDivDagSumLogFalse {
    using ConstZero = MAKE_CONST(PromteT, 0.0);
    using ConstOne = MAKE_CONST(PromteT, 1.0);
    using OpCopyInput = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyTarget = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyInputCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInput>;
    using OpCopyTargetCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyTarget>;
    using OpCompareMask = Bind<Vec::Compare<uint8_t, PromteT, COMPARE_MODE_NE>, OpCopyTargetCast, ConstZero>;
    using OpSelect = Bind<Vec::Select<uint8_t, PromteT, SELECT_MODE_T_S>, OpCompareMask, OpCopyTargetCast, ConstOne>;
    using OpLogTarget = Bind<Vec::Log<PromteT>, OpSelect>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpLogTarget, OpCopyInputCast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpCopyTargetCast, OpSub>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using OpCast = Bind<Vec::Cast<T, PromteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 'mean'和'batchmean',log_target为FALSE的DAG图描述
template <typename T, typename PromteT>
struct KLDivDagMeanLogFalse {
    using ConstZero = MAKE_CONST(PromteT, 0.0);
    using ConstOne = MAKE_CONST(PromteT, 1.0);
    using OpCopyInput = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyTarget = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyInputCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInput>;
    using OpCopyTargetCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyTarget>;
    using OpCompareMask = Bind<Vec::Compare<uint8_t, PromteT, COMPARE_MODE_NE>, OpCopyTargetCast, ConstZero>;
    using OpSelect = Bind<Vec::Select<uint8_t, PromteT, SELECT_MODE_T_S>, OpCompareMask, OpCopyTargetCast, ConstOne>;
    using OpLogTarget = Bind<Vec::Log<PromteT>, OpSelect>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpLogTarget, OpCopyInputCast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpCopyTargetCast, OpSub>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using OpMul1 = Bind<Vec::Muls<PromteT>, ReduceOp0, Placeholder::Var<PromteT, 0>>;
    using OpCast = Bind<Vec::Cast<T, PromteT, 1>, OpMul1>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 'sum'和'none',log_target为TRUE的DAG图描述
template <typename T, typename PromteT>
struct KLDivDagSumLogTrue {
    using OpCopyInput = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyTarget = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyInputCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInput>;
    using OpCopyTargetCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyTarget>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyTargetCast, OpCopyInputCast>;
    using OpExp = Bind<Vec::Exp<PromteT>, OpCopyTargetCast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpExp, OpSub>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using OpCast = Bind<Vec::Cast<T, PromteT, CAST_MODE_RINT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 'mean'和'batchmean',log_target为TRUE的DAG图描述
template <typename T, typename PromteT>
struct KLDivDagMeanLogTrue {
    using OpCopyInput = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyTarget = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyInputCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyInput>;
    using OpCopyTargetCast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyTarget>;
    using OpSub = Bind<Vec::Sub<PromteT>, OpCopyTargetCast, OpCopyInputCast>;
    using OpExp = Bind<Vec::Exp<PromteT>, OpCopyTargetCast>;
    using OpMul = Bind<Vec::Mul<PromteT>, OpExp, OpSub>;
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
    using OpMul1 = Bind<Vec::Muls<PromteT>, ReduceOp0, Placeholder::Var<PromteT, 0>>;
    using OpCast = Bind<Vec::Cast<T, PromteT, 1>, OpMul1>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace KLDivV2

#endif