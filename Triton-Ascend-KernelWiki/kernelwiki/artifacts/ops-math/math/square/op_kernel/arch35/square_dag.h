/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file square_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_SQUARE_DAG_H
#define CANN_CUSTOM_OPS_SQUARE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

template <typename T>
struct SquareOp {
    // 通过Compute构造计算图
    // y = x * x
    using OpCopyIn = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpSquare = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpCopyIn, OpCopyIn>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpSquare>;
    // 指定输出节点
    using Outputs = Ops::Base::Elems<OpCopyOut>; // 设置输出
    using OpDag = Ops::Base::DAGSch<Outputs>;
};

#endif // CANN_CUSTOM_OPS_SQUARE_DAG_H