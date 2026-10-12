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
 * \file is_neg_inf_dag.h
 * \brief
 */
#ifndef CANN_CUSTOM_OPS_IS_NEG_INF_DAG_H
#define CANN_CUSTOM_OPS_IS_NEG_INF_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace IsNegInfOp {
using namespace Ops::Base;
constexpr int32_t CAST_MODE = 0;
constexpr int32_t CMP_MODE = 2;
constexpr int32_t SELECT_MODE = 1;
constexpr int64_t ASCEND_API_BUFFER = 122880;  // 120K
constexpr int64_t ASCEND_WORKSPACE = 16777216; // 16M
constexpr float CONST_NEG_INF = -INFINITY;

template <typename U, typename T = float>
struct IsNegInfDAG {
    using ConstOne = MAKE_CONST(uint8_t, 1);
    using ConstZero = MAKE_CONST(uint8_t, 0);
    using ConstNegInf = MAKE_CONST(T, CONST_NEG_INF);
    using DataOne = Bind<Vec::Duplicate<uint8_t>, ConstOne>;

    using OpCopyIn = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCast = Bind<Vec::Cast<T, U, CAST_MODE>, OpCopyIn>;
    using CompareMask = Bind<Vec::Compare<uint8_t, T, CMP_MODE>, OpCast, ConstNegInf>;
    using SelectRes = Bind<Vec::Select<uint8_t, uint8_t, SELECT_MODE>, CompareMask, DataOne, ConstZero>;
    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, SelectRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace IsNegInfOp
#endif // CANN_CUSTOM_OPS_IS_NEG_INF_DAG_H