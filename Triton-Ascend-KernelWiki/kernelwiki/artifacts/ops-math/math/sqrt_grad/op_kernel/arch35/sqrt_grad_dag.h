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
 * \file sqrt_grad_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_SQRT_GRAD_DAG_H
#define CANN_CUSTOM_OPS_SQRT_GRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

const int COMPARE_MODE_NE = 5;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;

namespace SqrtGradDag {
template <typename U, typename T = float>
struct SqrtGradCustom {
    // 计算 0.5 * dy / y
    using ConstValue = MAKE_CONST(float, 0.5);
    using ConstValue1 = MAKE_CONST(float, 0.0);

    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyIn1 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In1<U>>;

    using OpCopyIn0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpCopyIn1Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn1>;
    using OpCopyIn1CastMul = Ops::Base::Bind<Ops::Base::Vec::Muls<T>, OpCopyIn1Cast, ConstValue>;
    using OpCompare =
        Ops::Base::Bind<Ops::Base::Vec::Compare<uint8_t, T, COMPARE_MODE_NE>, OpCopyIn1CastMul, ConstValue1>;
    using OpDiv = Ops::Base::Bind<Ops::Base::Vec::Div<T>, OpCopyIn1CastMul, OpCopyIn0Cast>;
    using OpSelect = Ops::Base::Bind<Ops::Base::Vec::Select<uint8_t, T, 1>, OpCompare, OpDiv, ConstValue1>;
    using OpResultCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, OpSelect>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
} // namespace SqrtGradDag
#endif // CANN_CUSTOM_OPS_SQRT_GRAD_DAG_H