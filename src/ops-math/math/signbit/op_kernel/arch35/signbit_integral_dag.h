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
 * \file signbit_integral_dag.h
 * \brief signbit_integral dag
 */

#ifndef SIGNBIT_INTEGRAL_DAG_H
#define SIGNBIT_INTEGRAL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace SignbitIntegralOp {
const int LESS_CMP_MODE = 0;
const int LESS_SELECT_MODE = 2;

template <typename T>
struct SignbitIntegralCompute {
    // 通过Compute构造计算图
    using Input0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using ConstZeroT = MAKE_CONST(T, 0);
    using DupZeroT = Bind<Vec::Duplicate<T>, ConstZeroT>;

    using ResCmp = Bind<Vec::Compare<uint8_t, T, LESS_CMP_MODE>, Input0, DupZeroT>;

    using ConstOne = MAKE_CONST(uint8_t, 1);
    using ConstZero = MAKE_CONST(uint8_t, 0);
    using DupOne = Bind<Vec::Duplicate<uint8_t>, ConstOne>;
    using DupZero = Bind<Vec::Duplicate<uint8_t>, ConstZero>;
    using ResSelect = Bind<Vec::Select<uint8_t, uint8_t, LESS_SELECT_MODE>, ResCmp, DupOne, DupZero>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, ResSelect>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace SignbitIntegralOp

#endif // SIGNBIT_INTEGRAL_DAG_H