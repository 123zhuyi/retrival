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
 * \file add_v2_dag.h
 * \brief add_v2 dag for ascend950 (arch35)
 */

#ifndef ADD_V2_DAG_H
#define ADD_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AddV2Op {
using namespace Ops::Base;

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
} // namespace AddV2Op

#endif // ADD_V2_DAG_H
