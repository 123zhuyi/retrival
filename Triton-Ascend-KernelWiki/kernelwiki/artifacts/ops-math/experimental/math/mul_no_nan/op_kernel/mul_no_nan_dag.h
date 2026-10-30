/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mul_no_nan_dag.h
 * \brief MulNoNan DAG computation graph (atvoss Broadcast mode)
 *
 * Computation: z = (y == 0) ? 0 : x * y
 *
 * Data flow:
 * x (GM) -> CopyInBrc --------\
 *                              -> Mul(x, y) -> mulResult -----\
 * y (GM) -> CopyInBrc --------/                                 \
 *                    |                                           -> Select(mask, mulResult, 0_scalar) -> CopyOut -> z
 * (GM)
 *                    +-> Compare(y, 0, NE) -> mask (bit=1:y!=0) /
 */

#ifndef MUL_NO_NAN_DAG_H
#define MUL_NO_NAN_DAG_H

#ifndef __CCE_AICORE__
#ifndef __aicore__
#define __aicore__
#endif
#endif

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace NsMulNoNan {

// Vec::Compare cmpMode: NE（not-equal）模板参数取值
constexpr uint8_t CMP_MODE_NE = 5;
// Vec::Select selMode: VSEL_TENSOR_SCALAR_MODE（src0 为 tensor，src1 为 scalar）
constexpr uint8_t SEL_MODE_TENSOR_SCALAR = 1;

template <typename T>
struct MulNoNanCompute {
    // CopyInBrc: GM -> UB with broadcast alignment
    using OpInputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputY = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    // Compute x * y
    using OpMulRes = Bind<Vec::Mul<T>, OpInputX, OpInputY>;

    // Compare y != 0; mask: bit=1 when y!=0, bit=0 when y==0
    using ConstZeroForCmp = MAKE_CONST(T, 0);
    using OpCmpMask = Bind<Vec::Compare<uint8_t, T, CMP_MODE_NE>, OpInputY, ConstZeroForCmp>;

    // NPU vsel: bit=1 -> src0, bit=0 -> src1
    // 期望 y!=0 -> mulResult，y==0 -> 0
    // 因此 src0 = OpMulRes（tensor），src1 = 0（scalar）
    using ConstZeroForSel = MAKE_CONST(T, 0);
    using OpSelectRes = Bind<Vec::Select<uint8_t, T, SEL_MODE_TENSOR_SCALAR>, OpCmpMask, OpMulRes, ConstZeroForSel>;

    // CopyOut: UB -> GM
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelectRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace NsMulNoNan

#endif // MUL_NO_NAN_DAG_H
