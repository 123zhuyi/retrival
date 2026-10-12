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
 * \file squared_difference_dag.h
 * \brief squared_difference dag
 */

#ifndef SQUARED_DIFFERENCE_DAG_H
#define SQUARED_DIFFERENCE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
namespace SquaredDifferenceOp {

using namespace Ops::Base;

template <typename T>
struct SquaredDifference {
    // 通过Compute构造计算图
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using DifferenceOut = Bind<Vec::Sub<T>, InputX1, InputX2>;

    using SquaredOut = Bind<Vec::Mul<T>, DifferenceOut, DifferenceOut>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SquaredOut>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct SquaredDifferenceCast {
    // 通过Compute构造计算图
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using InputCastX1 = Bind<Vec::Cast<float, T, 0>, InputX1>;
    using InputCastX2 = Bind<Vec::Cast<float, T, 0>, InputX2>;

    using DifferenceOut = Bind<Vec::Sub<float>, InputCastX1, InputCastX2>;

    using SquaredOut = Bind<Vec::Mul<float>, DifferenceOut, DifferenceOut>;

    using SquaredOutEnd = Bind<Vec::Cast<T, float, 1>, SquaredOut>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SquaredOutEnd>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace SquaredDifferenceOp
#endif // SQUARED_DIFFERENCE_DAG_H
