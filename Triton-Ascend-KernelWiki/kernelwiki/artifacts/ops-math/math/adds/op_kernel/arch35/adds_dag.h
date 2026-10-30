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
 * \file adds_dag.h
 * \brief Adds 算子 DAG 计算图定义（atvoss 框架 - Elewise 模式）
 *
 * 计算公式: output = input + scalar
 *
 * 数据流（bf16/fp16/int16/int32/int64）:
 * input (GM) -> CopyIn<T> -> Cast<T,float> -> Adds<float,scalar> -> Cast<float,T> -> CopyOut<T> -> output (GM)
 * 
 * 数据流（fp32）:
 * input (GM) -> CopyIn<float> -> Adds<float,scalar> -> CopyOut<float> -> output (GM)
 * （Cast fp32→fp32会被模板优化掉）
 * 
 * 标量参数通过 Placeholder::Var<float, 0> 占位，Kernel 入口通过 SetVar<float, 0>(value) 注入
 */

#ifndef ADDS_DAG_H
#define ADDS_DAG_H

// Host 编译时 mock __aicore__（Kernel 编译器已内置定义）
#ifndef __CCE_AICORE__
#ifndef __aicore__
#define __aicore__
#endif
#endif

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace NsAdds {

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
constexpr int CAST_MODE_ROUND = 4;
constexpr int CAST_MODE_TRUNC = 5;
constexpr int8_t REG_BIT_59 = 59;

// int16 特殊处理：使用截断模式（和 torch 保持一致）
template <class T1, class T2>
struct CastFp32ToInt16 : public Vec::ElemwiseUnaryOP<int16_t, float> {
    __aicore__ inline CastFp32ToInt16(LocalTensor<T1>& dst, LocalTensor<T2>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::SetCtrlSpr<REG_BIT_59, REG_BIT_59>(1);
        AscendC::Cast(dst, src, RoundMode::CAST_TRUNC, count);
        AscendC::SetCtrlSpr<REG_BIT_59, REG_BIT_59>(0);
#endif
    }
};

struct AddsInt16Op {
    using InputX = Bind<Vec::CopyIn<int16_t>, Placeholder::In0<int16_t>>;
    using CastX = Bind<Vec::Cast<float, int16_t, CAST_MODE_NONE>, InputX>;
    using Y = Bind<Vec::Adds<float>, CastX, Placeholder::Var<float, 0>>;
    using CastY = Bind<CastFp32ToInt16<int16_t, float>, Y>;
    using OpCopyOut = Bind<Vec::CopyOut<int16_t>, Placeholder::Out0<int16_t>, CastY>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 模板类型支持 float16, bfloat16, int32, float32
// Cast fp32到fp32场景，模板接口判断类型相等会优化掉cast逻辑
template <typename T, int castMode1, int castMode2>
struct AddsOp {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastX = Bind<Vec::Cast<float, T, castMode1>, InputX>;
    using Y = Bind<Vec::Adds<float>, CastX, Placeholder::Var<float, 0>>;
    using CastY = Bind<Vec::Cast<T, float, castMode2>, Y>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastY>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace NsAdds

#endif  // ADDS_DAG_H