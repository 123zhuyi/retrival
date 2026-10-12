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
 * \file sub_dag.h
 * \brief sub dag
 */

#ifndef SUB_DAG_H
#define SUB_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SubOp {
    using namespace AscendC;
    using namespace Ops::Base;
    constexpr int CAST_NONE_MODE = 0;
    constexpr int CAST_RINT_MODE = 1;

    template <typename T>
    struct SubWithoutCastCompute {
        using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
        using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

        using OpSubRes = Bind<Vec::Sub<T>, OpInputX1, OpInputX2>;

        using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSubRes>;

        using Outputs = Elems<OpCopyOut>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };

    template <typename T>
    struct SubWithCastCompute {
        using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
        using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

        using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
        using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;
        using OpSubRes = Bind<Vec::Sub<float>, OpCastX1, OpCastX2>;
        using OpCastRes = Bind<Vec::Cast<T, float, CAST_RINT_MODE>, OpSubRes>;

        using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

        using Outputs = Elems<OpCopyOut>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };

    template <typename T>
    struct SubBoolCompute {
        using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
        using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

        using OpSubRes = Bind<Vec::Sub<T>, OpInputX1, OpInputX2>;
        using OpAbsRes = Bind<Vec::Abs<T>, OpSubRes>;

        using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpAbsRes>;

        using Outputs = Elems<OpCopyOut>;
        using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
        using OpDag = DAGSch<Outputs, void, MemCfg>;
    };
}

#endif // SUB_DAG_H
