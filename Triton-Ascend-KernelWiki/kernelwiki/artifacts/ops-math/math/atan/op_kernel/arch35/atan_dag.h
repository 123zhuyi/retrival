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
 * \file atan_dag.h
 * atan:
 * atan(x) = x - x^3/3 + x^5/5 - x^7/7 + x^9/9 - x^11/11 + x^13/13... (|x|<=1)
 *         = pi/4 + atan((x-1)/(x+1)) (x>1)
 *         = pi/8 + atan((x-tan(pi/8))/(1+tan(pi/8)*x)) (tan(pi/8)=0.4142135623730950) (x > tan(pi/8) and x <
 * tan(pi/4)))
 */

#ifndef ATAN_DAG_H
#define ATAN_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AtanOp {
using namespace Ops::Base;

template <typename T>
struct AtanDag {
    using InputX1T = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    // cast
    using InputX1 = Bind<Vec::Cast<float, T, 0>, InputX1T>;

    using AtanRes = Bind<Vec::AtanPolyApprox<float>, InputX1>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, AtanRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace AtanOp

#endif // ATAN_DAG_H