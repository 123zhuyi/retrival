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
 * \file Ger_dag.h
 * \brief Ger dag
 */

#ifndef GER_DAG_H
#define GER_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace GerDag {
using namespace AscendC;
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;

// 支持float16, bfloat16 float, 升精度到float计算
template <typename T>
struct GerOp {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using CastX1 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputX2>;

    using Y = Bind<Vec::Mul<float>, CastX1, CastX2>;
    using YB16 = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, YB16>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace GerDag
#endif // GER_DAG_H