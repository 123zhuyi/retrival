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
 * \file div_no_nan_dag.h
 * \brief div_no_nan dag
 */

#ifndef DIV_NO_NAN_DAG_H
#define DIV_NO_NAN_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace DivNoNanOp {
using namespace AscendC;
using namespace Ops::Base;
constexpr int32_t CAST_MODE_NONE = 0;
constexpr int32_t CAST_MODE_RINT = 1;
constexpr int32_t CMP_MODE_NE = 5;
constexpr int32_t SEL_MODE = 2;

template <typename T>
struct DivNoNan {
    // 通过Compute构造计算图
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using ConstZero = MAKE_CONST(T, 0);
    using DataZero = Bind<Vec::Duplicate<T>, ConstZero>;

    using CmpRes = Bind<Vec::Compare<uint8_t, T, CMP_MODE_NE>, InputX2, DataZero>;

    using DivRes = Bind<Vec::Div<T>, InputX1, InputX2>;

    using SelectRes = Bind<Vec::Select<uint8_t, T, SEL_MODE>, CmpRes, DivRes, DataZero>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 输入数据是int8类型
template <typename T, typename PromoteT>
struct DivNoNanDagS8 {
    // 通过Compute构造计算图
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputX1Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX1>;
    using InputX2Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX2>;

    using ConstZero = MAKE_CONST(PromoteT, 0);
    using DataZero = Bind<Vec::Duplicate<PromoteT>, ConstZero>;

    using CmpRes = Bind<Vec::Compare<uint8_t, PromoteT, CMP_MODE_NE>, InputX2Cast, DataZero>;

    using DivRes = Bind<Vec::Div<PromoteT>, InputX1Cast, InputX2Cast>;

    using SelectRes = Bind<Vec::Select<uint8_t, PromoteT, SEL_MODE>, CmpRes, DivRes, DataZero>;
    using SelectResHalf = Bind<Vec::Cast<half, PromoteT, CAST_MODE_NONE>, SelectRes>;
    using SelectResInt = Bind<Vec::Cast<T, half, CAST_MODE_RINT>, SelectResHalf>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectResInt>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 输入数据是uint8类型->uint16_t
template <typename T, typename PromoteT>
struct DivNoNanU8Cast {
    // 通过Compute构造计算图
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputX1Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX1>;
    using InputX2Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX2>;

    using ConstZero = MAKE_CONST(PromoteT, 0);
    using DataZero = Bind<Vec::Duplicate<PromoteT>, ConstZero>;

    using CmpRes = Bind<Vec::Compare<uint8_t, PromoteT, CMP_MODE_NE>, InputX2Cast, DataZero>;

    using DivRes = Bind<Vec::Div<PromoteT>, InputX1Cast, InputX2Cast>;

    using SelectRes = Bind<Vec::Select<uint8_t, PromoteT, SEL_MODE>, CmpRes, DivRes, DataZero>;
    using SelectResCast = Bind<Vec::Cast<T, PromoteT, CAST_MODE_NONE>, SelectRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectResCast>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 输入数据是half或者bfloat16类型->float
template <typename T, typename PromoteT>
struct DivNoNanFloatCast {
    // 通过Compute构造计算图
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputX1Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX1>;
    using InputX2Cast = Bind<Vec::Cast<PromoteT, T, CAST_MODE_NONE>, InputX2>;

    using ConstZero = MAKE_CONST(PromoteT, 0);
    using DataZero = Bind<Vec::Duplicate<PromoteT>, ConstZero>;

    using CmpRes = Bind<Vec::Compare<uint8_t, PromoteT, CMP_MODE_NE>, InputX2Cast, DataZero>;

    using DivRes = Bind<Vec::Div<PromoteT>, InputX1Cast, InputX2Cast>;

    using SelectRes = Bind<Vec::Select<uint8_t, PromoteT, SEL_MODE>, CmpRes, DivRes, DataZero>;
    using SelectResCast = Bind<Vec::Cast<T, PromoteT, CAST_MODE_RINT>, SelectRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelectResCast>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace DivNoNanOp
#endif // DIV_NO_NAN_DAG_H