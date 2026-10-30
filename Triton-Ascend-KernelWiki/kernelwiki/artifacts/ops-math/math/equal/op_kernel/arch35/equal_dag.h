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
 * \file equal_dag.h
 * \brief equal dag
 */

#ifndef EQUAL_DAG_H
#define EQUAL_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace EqualOp {
using namespace AscendC;
using namespace Ops::Base;
const int CMP_MODE = 2;
const int SELECT_MODE = 2;

template <typename T>
struct EqualCompute {
    // 通过Compute构造计算图
    using ConstOne = MAKE_CONST(uint8_t, 1);
    using ConstZero = MAKE_CONST(uint8_t, 0);

    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using DataOne = Bind<Vec::Duplicate<uint8_t>, ConstOne>;
    using DataZero = Bind<Vec::Duplicate<uint8_t>, ConstZero>;

    using CompareMask = Bind<Vec::Compare<uint8_t, T, CMP_MODE>, InputX1, InputX2>;

    using SelectRes = Bind<Vec::Select<uint8_t, uint8_t, SELECT_MODE>, CompareMask, DataOne, DataZero>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, SelectRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace EqualOp
#endif // EQUAL_DAG_H
