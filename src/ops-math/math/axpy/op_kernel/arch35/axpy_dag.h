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
 * \file axpy_dag.h
 * \brief axpy dag
 */

#ifndef AXPY_DAG_H
#define AXPY_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AxpyOp
{
using namespace Ops::Base;

constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;
constexpr int CAST_TRUNC_MODE = 5;

// 模板类型支持int32, float16, bfloat16，float32
// Cast fp32到fp32场景，模板接口判断类型相等会优化掉cast逻辑
template <typename T, int castMode1, int castMode2>
struct AxpyCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpCastX1 = Bind<Vec::Cast<float, T, castMode1>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, castMode1>, OpInputX2>;
    using AxpyRes = Bind<Vec::Axpy<float>, OpCastX1, OpCastX2, Placeholder::Var<float, 0>>;
    using OpCastRes = Bind<Vec::Cast<T, float, castMode2>, AxpyRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct AxpyComputeFloat {
    using OpInputX1 = Bind<Vec::CopyInBrc<float>, Placeholder::In0<float>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<float>, Placeholder::In1<float>>;
    // Brc模板存在输入缓存场景，手动Copy规避Axpy计算结果覆盖OpInputX1
    using OpInputX1Copy = Bind<Vec::Copy<float>, OpInputX1>;
    using AxpyRes = Bind<Vec::Axpy<float>, OpInputX1Copy, OpInputX2, Placeholder::Var<float, 0>>;
    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, AxpyRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace AxpyOp
#endif  // AXPY_DAG_H