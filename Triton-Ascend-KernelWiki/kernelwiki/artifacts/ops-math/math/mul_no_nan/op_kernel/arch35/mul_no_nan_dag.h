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
 * \file mul_no_nan_dag.h
 * \brief mul_no_nan dag
 */

#ifndef MUL_NO_NAN_DAG_H
#define MUL_NO_NAN_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace MulNoNanOp {
using namespace AscendC;
using namespace Ops::Base;
constexpr int32_t CAST_MODE_NONE = 0;
constexpr int32_t CAST_MODE_RINT = 1;
// CMPMODE::NE = 5, see asc-devkit/impl/basic_api/utils/kernel_utils_mode.h.
constexpr int32_t CMP_MODE_NE = 5;
// SELMODE::VSEL_TENSOR_TENSOR_MODE = 2, semantics: dst = mask ? src0 : src1.
constexpr int32_t SEL_MODE = 2;

// Native path for dtypes that natively support Compare/Select/Mul without
// precision concerns: float, int32_t.
//   y = (x2 != 0) ? (x1 * x2) : 0
template <typename T>
struct MulNoNan {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using ConstZero = MAKE_CONST(T, 0);
    using DataZero = Bind<Vec::Duplicate<T>, ConstZero>;

    using CmpRes = Bind<Vec::Compare<uint8_t, T, CMP_MODE_NE>, InputX2, DataZero>;

    using MulRes = Bind<Vec::Mul<T>, InputX1, InputX2>;

    using SelectRes = Bind<Vec::Select<uint8_t, T, SEL_MODE>, CmpRes, MulRes, DataZero>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// Promoted path for half / bfloat16_t: cast inputs to PromoteT (float) for
// cmp/sel/mul to (a) match the DSL fallback that lifts to fp32 when fp16
// vcmpsel is unavailable, (b) avoid fp16 mid-range overflow before the
// mask is applied. Result is cast back to T with round-to-nearest-even.
template <typename T, typename PromoteT>
struct MulNoNanFloatCast {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputX1Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX1>;
    using InputX2Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX2>;

    using ConstZero = MAKE_CONST(PromoteT, 0);
    using DataZero = Bind<Vec::Duplicate<PromoteT>, ConstZero>;

    using CmpRes = Bind<Vec::Compare<uint8_t, PromoteT, CMP_MODE_NE>, InputX2Cast, DataZero>;

    using MulRes = Bind<Vec::Mul<PromoteT>, InputX1Cast, InputX2Cast>;

    using SelectRes = Bind<Vec::Select<uint8_t, PromoteT, SEL_MODE>, CmpRes, MulRes, DataZero>;
    using SelectResCast = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, SelectRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectResCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace MulNoNanOp
#endif // MUL_NO_NAN_DAG_H
