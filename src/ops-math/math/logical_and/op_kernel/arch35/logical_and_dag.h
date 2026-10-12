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
 * \file logical_and_dag.h
 * \brief
 */

#ifndef LOGICAL_AND_DAG_H
#define LOGICAL_AND_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"
namespace LogicalAndOp {
template <typename T>
struct LogicalAndCompute {
    // 通过Compute构造计算图
    using InputX1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using InputX2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;

    using LogicalAndRes = Ops::Base::Bind<Ops::Base::Vec::And<T>, InputX1, InputX2>;

    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, LogicalAndRes>;

    // 指定输出节点
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
} // namespace LogicalAndOp
#endif // LOGICAL_AND_DAG_H