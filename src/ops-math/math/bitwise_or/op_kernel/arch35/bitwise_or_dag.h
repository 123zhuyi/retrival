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
 * \file bitwise_or_dag.h
 * \brief bitwise or dag
 */

#ifndef BITWISE_OR_DAG_H
#define BITWISE_OR_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace BitwiseOrOp {
    using namespace AscendC;
    template <typename T>
    struct BitwiseOrCompute {
        using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
        using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

        using OpBitwiseOrRes = Bind<Vec::Or<T>, InputX1, InputX2>;

        using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpBitwiseOrRes>;

        using Outputs = Elems<OpCopyOut>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
}
#endif // BITWISE_OR_DAG_H