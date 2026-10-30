/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GREATER_DAG_H
#define GREATER_DAG_H


#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace GreaterOp {
    using namespace AscendC;
    using namespace Ops::Base;
    constexpr int GREATER_CMP_MODE = 1;
    constexpr int GREATER_SEL_MODE = 2;

    template <typename T>
    struct GreaterCompute {
        using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
        using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

        using CompareRes = Bind<Vec::Compare<uint8_t, T, GREATER_CMP_MODE>, InputX1, InputX2>;

        using ConstZero = MAKE_CONST(uint8_t, 0);
        using DupZero = Bind<Vec::Duplicate<uint8_t>, ConstZero>;
        using ConstOne = MAKE_CONST(uint8_t, 1);
        using DupOne = Bind<Vec::Duplicate<uint8_t>, ConstOne>;
        using SelectRes = Bind<Vec::Select<uint8_t, uint8_t, GREATER_SEL_MODE>, CompareRes, DupOne, DupZero>;

        using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, SelectRes>;
        using Outputs = Elems<OpCopyOut>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
}

#endif // GREATER_DAG_H