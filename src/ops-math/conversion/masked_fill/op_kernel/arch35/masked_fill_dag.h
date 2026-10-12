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
* \file masked_fill_dag.h
* \brief masked_fill dag
*/

#ifndef MASKED_FILL_DAG_H
#define MASKED_FILL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace AscendC;
using namespace Ops::Base;

template <typename T>
struct MaskedFill {
    // 输入张量
    using InputX = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    // 掩码张量（布尔类型）
    using InputMask = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;

    using ConstZero = MAKE_CONST(int8_t, 0);

    using CompareRes = Bind<Vec::Compare<uint8_t, int8_t, 2>, InputMask, ConstZero>; // 2表示CMPMODE::EQ
    using OpSelect = Bind<Vec::Select<uint8_t, T, 1>, CompareRes, InputX, Placeholder::In2<T, Placeholder::ScalarAttr<true>>>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpSelect>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

#endif // MASKED_FILL_DAG_H