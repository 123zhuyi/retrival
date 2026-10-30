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
 * \file add_dag.h
 * \brief add dag
 */

#ifndef ADD_DAG_H
#define ADD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
using namespace AscendC;

namespace AddOp {
constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;

template <typename T>
struct AddWithoutCastCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpAddRes = Bind<Vec::Add<T>, OpInputX1, OpInputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpAddRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AddWithCastCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;
    using OpAddRes = Bind<Vec::Add<float>, OpCastX1, OpCastX2>;
    using OpCastRes = Bind<Vec::Cast<T, float, CAST_RINT_MODE>, OpAddRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct AddMixDtypeCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T2>, Placeholder::In1<T2>>;

    using OpCastX1 = Bind<Vec::Cast<float, T1, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T2, CAST_NONE_MODE>, OpInputX2>;
    using OpAddRes = Bind<Vec::Add<float>, OpCastX1, OpCastX2>;

    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, OpAddRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AddBoolCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpAddRes = Bind<Vec::Add<T>, OpInputX1, OpInputX2>;
    using ConstOne = MAKE_CONST(T, 1);
    using OpMinsRes = Bind<Vec::Mins<T>, OpAddRes, ConstOne>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpMinsRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace AddOp

#endif // ADD_DAG_H
