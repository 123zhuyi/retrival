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
 * \file addcmul_dag.h
 * \brief
 */

#ifndef ADDCMUL_DAG_H
#define ADDCMUL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
namespace AddcmulOp {
constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;

template <typename T1, typename T2>
struct AddcmulFloatOp {
    using OpInputData = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T1>, Ops::Base::Placeholder::In0<T1>>;
    using OpInputX1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T1>, Ops::Base::Placeholder::In1<T1>>;
    using OpInputX2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T1>, Ops::Base::Placeholder::In2<T1>>;
    using OpInputValue = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T2>, Ops::Base::Placeholder::In3<T2>>;

    using OpCastInputData = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T1, CAST_NONE_MODE>, OpInputData>;
    using OpCastX1 = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T1, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T1, CAST_NONE_MODE>, OpInputX2>;
    using OpCastVal = Ops::Base::Bind<Ops::Base::Vec::Cast<float, T2, CAST_NONE_MODE>, OpInputValue>;

    using OpMulX1X2Res = Ops::Base::Bind<Ops::Base::Vec::Mul<float>, OpCastX1, OpCastX2>;
    using OpMulAddDstRes =
        Ops::Base::Bind<Ops::Base::Vec::FusedMulAdd<float>, OpCastInputData, OpMulX1X2Res, OpCastVal>;
    using OpCastRes = Ops::Base::Bind<Ops::Base::Vec::Cast<T1, float, CAST_RINT_MODE>, OpMulAddDstRes>;

    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T1>, Ops::Base::Placeholder::Out0<T1>, OpCastRes>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AddcmulInt32Op {
    using OpInputData = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In0<T>>;
    using OpInputX1 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In1<T>>;
    using OpInputX2 = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In2<T>>;
    using OpInputValue = Ops::Base::Bind<Ops::Base::Vec::CopyInBrc<T>, Ops::Base::Placeholder::In3<T>>;

    using OpMulX1X2Res = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpInputX1, OpInputX2>;
    using OpMulValRes = Ops::Base::Bind<Ops::Base::Vec::Mul<T>, OpMulX1X2Res, OpInputValue>;
    using OpAddRes = Ops::Base::Bind<Ops::Base::Vec::Add<T>, OpInputData, OpMulValRes>;

    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpAddRes>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
} // namespace AddcmulOp

#endif // ADDCMUL_DAG_H
