/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file logical_not_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_LOGICAL_NOT_DAG_H
#define CANN_CUSTOM_OPS_LOGICAL_NOT_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
namespace LogicalNotOp {
using namespace AscendC;
using namespace Ops::Base;
template <typename T>
struct LogicalNotDag {
    using ConstValue = MAKE_CONST(float, -1.0);

    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using OpCast0 = Bind<Vec::Cast<half, T, 0>, OpCopyIn0>;
    using OpAdds = Bind<Vec::Adds<half>, OpCast0, ConstValue>;
    using OpAbs = Bind<Vec::Abs<half>, OpAdds>;
    using OpCast1 = Bind<Vec::Cast<T, half, 0>, OpAbs>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCast1>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace LogicalNotOp

#endif // CANN_CUSTOM_OPS_LOGICAL_NOT_DAG_H