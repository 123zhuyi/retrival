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
 * \file real_div_dag.h
 * \brief real div dag
 */

#ifndef REAL_DIV_DAG_H
#define REAL_DIV_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace RealDivOp
{
using namespace AscendC;
using namespace Ops::Base;

constexpr int DIV_CAST_MODE_NONE = 0;
constexpr int DIV_CAST_MODE_RINT = 1;

template <typename T1>
struct RealDivIntegerWithoutCast {
    // int32
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivRes = Bind<Vec::Div<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct RealDivFloatWithoutCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivRes = Bind<Vec::DivHighPrecision<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct RealDivFloatWithCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivRes = Bind<Vec::DivHighPrecision<T2>, CastX1, CastX2>;
    using CastOut = Bind<Vec::Cast<T1, T2, DIV_CAST_MODE_RINT>, DivRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct RealDivWithBool {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1Half = Bind<Vec::Cast<half, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2Half = Bind<Vec::Cast<half, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using CastX1Float = Bind<Vec::Cast<float, half, DIV_CAST_MODE_NONE>, CastX1Half>;
    using CastX2Float = Bind<Vec::Cast<float, half, DIV_CAST_MODE_NONE>, CastX2Half>;
    using DivRes = Bind<Vec::Div<float>, CastX1Float, CastX2Float>;
    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, DivRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct RealDivIntegerWithCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<float, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<float, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivRes = Bind<Vec::DivHighPrecision<float>, CastX1, CastX2>;
    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, DivRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace RealDivOp

#endif  // REAL_DIV_DAG_H