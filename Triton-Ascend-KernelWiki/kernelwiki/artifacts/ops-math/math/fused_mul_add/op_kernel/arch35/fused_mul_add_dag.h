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
 * \file fused_mul_add_dag.h
 * \brief
 */

#ifndef FUSED_MUL_ADD_DAG_H
#define FUSED_MUL_ADD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace FusedMulAddOp {
constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;

// Float / Half path: cast all inputs to fp32, then compute x1*x2 + x3 as two
// explicit element-wise ops, finally cast the result back to T.
//
// IMPORTANT: do NOT use Vec::FusedMulAdd here. Its underlying implementation
// (AscendC::FusedMulAdd(src2, alpha, src1, count) followed by Copy(dst, src2))
// writes back into the src2 buffer in-place. When src2 is a CopyInBrc'd
// placeholder (e.g. OpCastX1) that atvoss reuses across tiles in a broadcast
// schedule, the second tile reads a corrupted x1, producing incorrect outputs
// (~50% / ~0% / ~-50% precision on huge broadcast cases observed on Ascend 950).
// Using explicit Mul + Add keeps every input read-only on its UB buffer.
template <typename T>
struct FusedMulAddFloatOp {
    using OpInputX1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using OpInputX2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
    using OpInputX3 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;

    using OpCastX1 = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;
    using OpCastX3 = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX3>;

    using OpMul     = Ops::Base::Bind<Ops::Base::Vec::Mul<float>, OpCastX1, OpCastX2>;
    using OpAdd     = Ops::Base::Bind<Ops::Base::Vec::Add<float>, OpMul, OpCastX3>;
    using OpCastRes = Ops::Base::Bind<Ops::Base::Vec::Cast<T, float, CAST_RINT_MODE>, OpAdd>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpCastRes>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg  = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag   = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

// Int32 path: no hardware FMA available, fall back to explicit Mul then Add.
template <typename T>
struct FusedMulAddInt32Op {
    using OpInputX1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using OpInputX2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
    using OpInputX3 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;

    using OpMul = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpInputX1, OpInputX2>;
    using OpAdd = Ops::Base::Bind<Ops::Base::Vec::Add<T>, OpMul, OpInputX3>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpAdd>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg  = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag   = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
} // namespace FusedMulAddOp

#endif // FUSED_MUL_ADD_DAG_H
