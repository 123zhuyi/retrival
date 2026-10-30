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
 * \file is_finite_dag.h
 * \brief
 */
#ifndef CANN_CUSTOM_OPS_IS_FINITE_DAG_H
#define CANN_CUSTOM_OPS_IS_FINITE_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace IsFiniteOp {
using namespace Ops::Base;
constexpr uint64_t TILING_KEY_FP16 = 101UL;
constexpr uint64_t TILING_KEY_BF16 = 102UL;
constexpr uint64_t TILING_KEY_FP32 = 103UL;
constexpr int64_t ASCEND_API_BUFFER = 122880;  // 120K
constexpr int64_t ASCEND_WORKSPACE = 16777216; // 16M
constexpr int32_t CAST_MODE_NONE = 0;
constexpr int32_t CMP_MODE_LT = 0;
constexpr int32_t SEL_MODE_TS = 1; // tensor scalar mode
constexpr float INF_CONST = INFINITY;

template <typename U, typename T = float>
struct IsFiniteDag {
    using ConstValueOne = MAKE_CONST(uint8_t, 1);
    using ConstValueZero = MAKE_CONST(uint8_t, 0);
    using ConstValueInfinify = MAKE_CONST(T, INF_CONST);
    using Ones = Bind<Vec::Duplicate<uint8_t>, ConstValueOne>;

    using OpCopyIn = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInCast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn>;
    using OpCopyInCastAbs = Bind<Vec::Abs<T>, OpCopyInCast>;
    using OpIsFiniteMask = Bind<Vec::Compare<uint8_t, T, CMP_MODE_LT>, OpCopyInCastAbs, ConstValueInfinify>;
    using OpSelect = Bind<Vec::Select<uint8_t, uint8_t, SEL_MODE_TS>, OpIsFiniteMask, Ones, ConstValueZero>;
    // copy out
    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, OpSelect>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace IsFiniteOp

#endif // CANN_CUSTOM_OPS_IS_FINITE_DAG_H