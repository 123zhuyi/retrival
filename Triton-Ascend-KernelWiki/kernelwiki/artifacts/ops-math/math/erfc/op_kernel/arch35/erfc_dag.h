/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file erfc_dag.h
 * \brief
 */

#ifndef OPS_MATH_ERFC_DAG_H
#define OPS_MATH_ERFC_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace ErfcOp {
using namespace Ops::Base;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

// =============================================================================
// DAG variant 1: float32 direct computation (no Cast)
// =============================================================================
template <typename T>
struct ErfcWithoutCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpResult = Bind<Vec::Erfc<float>, OpCopyIn>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// =============================================================================
// DAG variant 2: half/bfloat16 with Cast promotion to float32
// =============================================================================
template <typename T>
struct ErfcWithCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyInCast = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, OpCopyIn>;
    using OpResult = Bind<Vec::Erfc<float>, OpCopyInCast>;
    using OpResultCast = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, OpResult>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace ErfcOp

#endif // OPS_MATH_ERFC_DAG_H
