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
 * \file muls_dag.h
 * \brief muls dag
 */

#ifndef MULS_DAG_H
#define MULS_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace MulsDag
{
using namespace Ops::Base;

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
constexpr int CAST_MODE_ROUND = 4;
constexpr int CAST_MODE_TRUNC = 5;
constexpr uint32_t COUNT_DOUBLE = 2;
constexpr int8_t REG_BIT_59 = 59;

template <class T1, class T2>  // complex64, complex32
struct CastComplex32ToComplex64 : public Vec::ElemwiseUnaryOP<T1, T2> {
    __aicore__ inline CastComplex32ToComplex64(LocalTensor<T1>& dst, LocalTensor<T2>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::Cast(dst.template ReinterpretCast<float>(), src.template ReinterpretCast<half>(), RoundMode::CAST_NONE,
                      count * COUNT_DOUBLE);
#endif
    }
};
template <class T1, class T2>  // complex32, complex64
struct CastComplex64ToComplex32 : public Vec::ElemwiseUnaryOP<T1, T2> {
    __aicore__ inline CastComplex64ToComplex32(LocalTensor<T1>& dst, LocalTensor<T2>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        AscendC::Cast(dst.template ReinterpretCast<half>(), src.template ReinterpretCast<float>(), RoundMode::CAST_RINT,
                      count * COUNT_DOUBLE);
#endif
    }
};
template <class T1, class T2>  // complex64
struct MulsComplex64Float : public Vec::ElemwiseBinaryOP<T1, T1, T2> {
    __aicore__ inline MulsComplex64Float(const LocalTensor<T1>& dst, const LocalTensor<T1>& src, const T2& scalar,
                                         int count)
    {
#ifdef __CCE_AICORE__
        AscendC::Muls(dst.template ReinterpretCast<float>(), src.template ReinterpretCast<float>(),
                      static_cast<float>(scalar), count * COUNT_DOUBLE);
#endif
    }
};

template <class T1, class T2>  // int16_t, float
struct CastFp32ToInt16 : public Vec::ElemwiseUnaryOP<int16_t, float> {
    __aicore__ inline CastFp32ToInt16(LocalTensor<T1>& dst, LocalTensor<T2>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        // 设置成截断模式，和torch保持一致
        AscendC::SetCtrlSpr<REG_BIT_59, REG_BIT_59>(1);
        AscendC::Cast(dst, src, RoundMode::CAST_TRUNC, count);
        AscendC::SetCtrlSpr<REG_BIT_59, REG_BIT_59>(0);
#endif
    }
};

struct MulsInt16Op {
    using InputX = Bind<Vec::CopyIn<int16_t>, Placeholder::In0<int16_t>>;
    using CastX = Bind<Vec::Cast<float, int16_t, CAST_MODE_NONE>, InputX>;
    using Y = Bind<Vec::Muls<float>, CastX, Placeholder::Var<float, 0>>;
    using CastY = Bind<CastFp32ToInt16<int16_t, float>, Y>;
    using OpCopyOut = Bind<Vec::CopyOut<int16_t>, Placeholder::Out0<int16_t>, CastY>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 模板类型支持int32, int64, float16, bfloat16，float32
// Cast fp32到fp32场景，模板接口判断类型相等会优化掉cast逻辑
template <typename T, int castMode1, int castMode2>
struct MulsOp {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastX = Bind<Vec::Cast<float, T, castMode1>, InputX>;
    using Y = Bind<Vec::Muls<float>, CastX, Placeholder::Var<float, 0>>;
    using CastY = Bind<Vec::Cast<T, float, castMode2>, Y>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastY>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 模板类型传入complex32, complex64
template <typename T, typename PromoteT>
struct MulsComplex32Op {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using CastX = Bind<CastComplex32ToComplex64<PromoteT, T>, InputX>;
    using Y = Bind<MulsComplex64Float<PromoteT, float>, CastX, Placeholder::Var<float, 0>>;
    using CastY = Bind<CastComplex64ToComplex32<T, PromoteT>, Y>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastY>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// 模板类型传入complex64
template <typename T>
struct MulsComplex64Op {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Y = Bind<MulsComplex64Float<T, float>, InputX, Placeholder::Var<float, 0>>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
}  // namespace MulsDag
#endif  // MULS_DAG_H