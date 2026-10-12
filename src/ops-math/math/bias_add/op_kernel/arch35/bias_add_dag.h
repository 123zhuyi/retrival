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
 * \file bias_add_dag.h
 * \brief bias_add dag
 */

#ifndef BIAS_ADD_DAG_H
#define BIAS_ADD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace BiasAddOp {
using namespace AscendC;
using namespace Ops::Base;

template <typename T>
struct BiasAddCompute {
    // 通过Compute构造计算图
    using Value = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using Bias = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using ResAdd = Bind<Vec::Add<T>, Value, Bias>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, ResAdd>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace BiasAddOp

#endif // BIAS_ADD_DAG_H