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
 * \file axpy_v2_dag.h
 * \brief axpy_v2 dag
 */

#ifndef AXPY_V2_DAG_H
#define AXPY_V2_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AxpyV2Op
{
using namespace Ops::Base;

constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;
constexpr int CAST_ROUND_MODE = 4;
constexpr std::int32_t FF = 255;

template <class T>
struct AndFF : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline AndFF(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        Duplicate(dst, static_cast<T>(FF), count);
        And(dst, src, dst, count);
#endif
    }
};

template <typename T, int castMode1, int castMode2>
struct AxpyV2ComputeFp16Bf16 {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using OpCastX1 = Bind<Vec::Cast<float, T, castMode1>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, castMode1>, OpInputX2>;
    using OpCastAlpha = Bind<Vec::Cast<float, T, castMode1>, OpInputAlpha>;
    using FusedMulAddRes = Bind<Vec::FusedMulAdd<float>, OpCastX1, OpCastX2, OpCastAlpha>;
    using OpCastRes = Bind<Vec::Cast<T, float, castMode2>, FusedMulAddRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct AxpyV2ComputeFloat {
    using OpInputX1 = Bind<Vec::CopyInBrc<float>, Placeholder::In0<float>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<float>, Placeholder::In1<float>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<float>, Placeholder::In2<float>>;
    // Brc模板存在输入缓存场景，手动Copy规避FusedMulAdd计算结果覆盖OpInputX2
    using OpInputX2Copy = Bind<Vec::Copy<float>, OpInputX2>;
    using FusedMulAddRes = Bind<Vec::FusedMulAdd<float>, OpInputX1, OpInputX2Copy, OpInputAlpha>;
    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, FusedMulAddRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AxpyV2ComputeInt32Int64 {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using OpMulRes = Bind<Vec::Mul<T>, OpInputX2, OpInputAlpha>;
    using OpAddRes = Bind<Vec::Add<T>, OpInputX1, OpMulRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpAddRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct AxpyV2ComputeUint8Int8 {
    using OpInputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<T1>, Placeholder::In2<T1>>;
    using OpCastX2 = Bind<Vec::Cast<T2, T1, CAST_NONE_MODE>, OpInputX2>;
    using OpCastAlpha = Bind<Vec::Cast<T2, T1, CAST_NONE_MODE>, OpInputAlpha>;
    using Y = Bind<Vec::Mul<T2>, OpCastX2, OpCastAlpha>;
    using Y1 = Bind<AndFF<int32_t>, Y>;
    using Y2 = Bind<Vec::Cast<uint8_t, T2, CAST_NONE_MODE>, Y1>;
    using Res = Bind<Vec::Add<T1>, OpInputX1, Y2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, Res>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

struct AxpyV2ComputeBool {
    using ConstOne = MAKE_CONST(float, 1.0f);
    using OpInputX1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using OpInputAlpha = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In2<int8_t>>;
    using OpCastX1 = Bind<Vec::Cast<half, int8_t, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<half, int8_t, CAST_NONE_MODE>, OpInputX2>;
    using OpCastAlpha = Bind<Vec::Cast<half, int8_t, CAST_NONE_MODE>, OpInputAlpha>;
    using FusedMulAddRes = Bind<Vec::FusedMulAdd<half>, OpCastX1, OpCastX2, OpCastAlpha>;
    using MinRes = Bind<Vec::Mins<half>, FusedMulAddRes, ConstOne>;
    using Res = Bind<Vec::Cast<int8_t, half, CAST_ROUND_MODE>, MinRes>;
    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, Res>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}  // namespace AxpyV2Op
#endif  // AXPY_V2_DAG_H