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
 * \file log_add_exp_dag.h
 * \brief log_add_exp dag
 *
 * Simplified (base=-1, scale=1.0, shift=0.0):
 *   y = max(x1, x2) + ln(1 + exp(-|x1 - x2|))
 *
 * Full (any non-default):
 *   base=-1: y = max(x1, x2) + ln(1 + exp((-|x1 - x2|) * scale + shift))
 *   base>0:  y = max(x1, x2) + ln(1 + exp(((-|x1-x2|)*scale+shift)*ln(base))) / ln(base)
 */

#ifndef LOG_ADD_EXP_DAG_H
#define LOG_ADD_EXP_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;
using namespace AscendC;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace LogAddExpOp {
constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;
constexpr int CMP_EQ_MODE = 2;   // AscendC::CMPMODE::EQ
constexpr int CMP_NE_MODE = 5;   // AscendC::CMPMODE::NE
constexpr int SEL_TT_MODE = 2;   // AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE
constexpr float POS_INF = INFINITY;
constexpr float NEG_INF = -INFINITY;

// inf 修正：x1、x2 同为 ±inf 时 x1-x2=NaN，将差值置 0，使结果回到 max+ln(2)=±inf。
template <typename CT, typename In1, typename In2>
struct InfGuardedSub {
    using OpMax = Bind<Vec::Max<CT>, In1, In2>;
    using OpMin = Bind<Vec::Min<CT>, In1, In2>;
    using OpSub = Bind<Vec::Sub<CT>, In1, In2>;
    using ConstZero = MAKE_CONST(CT, 0);
    using DupZero = Bind<Vec::Duplicate<CT>, ConstZero>;
    using ConstPosInf = MAKE_CONST(CT, POS_INF);
    using ConstNegInf = MAKE_CONST(CT, NEG_INF);
    using MaskBothPos = Bind<Vec::Compare<uint8_t, CT, CMP_EQ_MODE>, OpMin, ConstPosInf>;
    using SubFixPos = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskBothPos, DupZero, OpSub>;
    using MaskBothNeg = Bind<Vec::Compare<uint8_t, CT, CMP_EQ_MODE>, OpMax, ConstNegInf>;
    using SubFixed = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskBothNeg, DupZero, SubFixPos>;
};

// 稳定计算 log1p(x)，避免 x 很小时 1+x 舍入为 1 导致结果变 0。
template <typename CT, typename In>
struct StableLog1p {
    using ConstOne = MAKE_CONST(CT, 1);
    using ConstNegOne = MAKE_CONST(CT, -1);
    using OpAddOne = Bind<Vec::Adds<CT>, In, ConstOne>;
    using OpMid = Bind<Vec::Adds<CT>, OpAddOne, ConstNegOne>;
    using OpRatio = Bind<Vec::Div<CT>, In, OpMid>;
    using OpLog = Bind<Vec::Log<CT>, OpAddOne>;
    using OpMul = Bind<Vec::Mul<CT>, OpLog, OpRatio>;
    using MaskNotOne = Bind<Vec::Compare<uint8_t, CT, CMP_NE_MODE>, OpAddOne, ConstOne>;
    using FixSmall = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskNotOne, OpMul, In>;
    using ConstPosInf = MAKE_CONST(CT, POS_INF);
    using DupPosInf = Bind<Vec::Duplicate<CT>, ConstPosInf>;
    using MaskNotInf = Bind<Vec::Compare<uint8_t, CT, CMP_NE_MODE>, OpAddOne, ConstPosInf>;
    using OpOut = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskNotInf, FixSmall, DupPosInf>;
};

// ==================== Simplified (base=-1, scale=1.0, shift=0.0) ====================

template <typename T>
struct LogAddExpSimplifiedCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using Guard = InfGuardedSub<T, OpInputX1, OpInputX2>;
    using OpMax = typename Guard::OpMax;
    using OpAbs = Bind<Vec::Abs<T>, typename Guard::SubFixed>;
    using ConstNegOne = MAKE_CONST(T, -1);
    using OpNeg = Bind<Vec::Muls<T>, OpAbs, ConstNegOne>;
    using OpExp = Bind<Vec::Exp<T>, OpNeg>;
    using OpLog = typename StableLog1p<T, OpExp>::OpOut;
    using OpAdd = Bind<Vec::Add<T>, OpMax, OpLog>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpAdd>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LogAddExpSimplifiedWithCastCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;

    using Guard = InfGuardedSub<float, OpCastX1, OpCastX2>;
    using OpMax = typename Guard::OpMax;
    using OpAbs = Bind<Vec::Abs<float>, typename Guard::SubFixed>;
    using ConstNegOne = MAKE_CONST(float, -1);
    using OpNeg = Bind<Vec::Muls<float>, OpAbs, ConstNegOne>;
    using OpExp = Bind<Vec::Exp<float>, OpNeg>;
    using OpLog = typename StableLog1p<float, OpExp>::OpOut;
    using OpAdd = Bind<Vec::Add<float>, OpMax, OpLog>;

    using OpCastRes = Bind<Vec::Cast<T, float, CAST_RINT_MODE>, OpAdd>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ==================== Full (with base, scale, shift) ====================
// Pos(=SetScalar 顺序): 0=negScale, 1=shift, 2=lnBase, 3=invLnBase

template <typename T>
struct LogAddExpFullCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using Guard = InfGuardedSub<T, OpInputX1, OpInputX2>;
    using OpMax = typename Guard::OpMax;
    using OpAbs = Bind<Vec::Abs<T>, typename Guard::SubFixed>;
    using VarNegScale = Placeholder::Var<float, 0>;
    using OpNegScale = Bind<Vec::Muls<T>, OpAbs, VarNegScale>;
    using VarShift = Placeholder::Var<float, 1>;
    using OpShift = Bind<Vec::Adds<T>, OpNegScale, VarShift>;
    using VarLnBase = Placeholder::Var<float, 2>;
    using OpMulLnBase = Bind<Vec::Muls<T>, OpShift, VarLnBase>;
    using OpExp = Bind<Vec::Exp<T>, OpMulLnBase>;
    using OpLog = typename StableLog1p<T, OpExp>::OpOut;
    using VarInvLnBase = Placeholder::Var<float, 3>;
    using OpMulInvLnBase = Bind<Vec::Muls<T>, OpLog, VarInvLnBase>;
    using OpAdd = Bind<Vec::Add<T>, OpMax, OpMulInvLnBase>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpAdd>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct LogAddExpFullWithCastCompute {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;

    using Guard = InfGuardedSub<float, OpCastX1, OpCastX2>;
    using OpMax = typename Guard::OpMax;
    using OpAbs = Bind<Vec::Abs<float>, typename Guard::SubFixed>;
    using VarNegScale = Placeholder::Var<float, 0>;
    using OpNegScale = Bind<Vec::Muls<float>, OpAbs, VarNegScale>;
    using VarShift = Placeholder::Var<float, 1>;
    using OpShift = Bind<Vec::Adds<float>, OpNegScale, VarShift>;
    using VarLnBase = Placeholder::Var<float, 2>;
    using OpMulLnBase = Bind<Vec::Muls<float>, OpShift, VarLnBase>;
    using OpExp = Bind<Vec::Exp<float>, OpMulLnBase>;
    using OpLog = typename StableLog1p<float, OpExp>::OpOut;
    using VarInvLnBase = Placeholder::Var<float, 3>;
    using OpMulInvLnBase = Bind<Vec::Muls<float>, OpLog, VarInvLnBase>;
    using OpAdd = Bind<Vec::Add<float>, OpMax, OpMulInvLnBase>;

    using OpCastRes = Bind<Vec::Cast<T, float, CAST_RINT_MODE>, OpAdd>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace LogAddExpOp

#endif // LOG_ADD_EXP_DAG_H
