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
 * \file div_dag.h
 * \brief div dag
 */

#ifndef DIV_DAG_H
#define DIV_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace DivOp {
using namespace Ops::Base;
constexpr int DIV_CAST_MODE_NONE = 0;
constexpr int DIV_CAST_MODE_RINT = 1;
constexpr int DIV_CMP_NE_MODE = 5;
constexpr int DIV_SEL_TENSOR_TENSOR_MODE = 2;
constexpr int8_t SAT_POS = 60;
constexpr int16_t DIV_B16_SIGN = -32768;
constexpr int32_t DIV_B32_SIGN = -2147483648;

template <class R, class T, int roundMode>
struct CastOverFlow : public Vec::ElemwiseUnaryOP<R, T> {
    __aicore__ inline CastOverFlow(LocalTensor<R>& dst, LocalTensor<T>& src, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        SetCtrlSpr<SAT_POS, SAT_POS>(0);
        constexpr static Reg::CastTrait castTrait3 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> vreg0;
            Reg::RegTensor<R> vreg1;
            Reg::MaskReg preg0;
            // sizeof(T) must > sizeof(R)
            uint32_t size = count;
            uint16_t vfLoopNum = (size + (VECTOR_REG_WIDTH / sizeof(T)) - 1) / (VECTOR_REG_WIDTH / sizeof(T));
            __local_mem__ T* bufferIn0Addr = (__local_mem__ T*)src.GetPhyAddr();
            __local_mem__ R* bufferOut0Addr = (__local_mem__ R*)dst.GetPhyAddr();
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg0 = Reg::UpdateMask<T>(size);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(vreg0, bufferIn0Addr + i * (VECTOR_REG_WIDTH / sizeof(T)));
                Reg::Cast<R, T, castTrait3>(vreg1, vreg0, preg0);
                Reg::DataCopy<R, Reg::StoreDist::DIST_PACK_B16>(bufferOut0Addr + i * (VECTOR_REG_WIDTH / sizeof(T)),
                                                                vreg1, preg0);
            }
        }
        SetCtrlSpr<SAT_POS, SAT_POS>(1);
#endif
    }
};

template <typename T1>
struct DivComplexWithoutCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivRes = Bind<Vec::Div<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct DivFloatWithoutCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivRes = Bind<Vec::DivHighPrecision<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct DivFloatWithCast {
    // half and bfloat16
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivRes = Bind<Vec::DivHighPrecision<T2>, CastX1, CastX2>;
    using CastOut = Bind<Vec::Cast<T1, T2, DIV_CAST_MODE_RINT>, DivRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct DivIntegerS8 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::Div<T2>, CastX1, CastX2>;
    using MulValue = Bind<Vec::Mul<T2>, CastX2, DivValue>;
    using SubValue = Bind<Vec::Sub<T2>, CastX1, MulValue>;
    using ConstZero = MAKE_CONST(T2, 0);
    using DupZero = Bind<Vec::Duplicate<T2>, ConstZero>;
    using RemMask = Bind<Vec::Compare<uint8_t, T2, DIV_CMP_NE_MODE>, SubValue, DupZero>;
    using ConstFlag = MAKE_CONST(T2, DIV_B16_SIGN);
    using DupFlag = Bind<Vec::Duplicate<T2>, ConstFlag>;
    using AndX1 = Bind<Vec::And<T2>, CastX1, DupFlag>;
    using AndX2 = Bind<Vec::And<T2>, CastX2, DupFlag>;
    using SignMask = Bind<Vec::Compare<uint8_t, T2, DIV_CMP_NE_MODE>, AndX1, AndX2>;
    using ResMask = Bind<Vec::And<uint8_t>, RemMask, SignMask>;
    using ConstOne = MAKE_CONST(T2, 1);
    using DupOne = Bind<Vec::Duplicate<T2>, ConstOne>;
    using SubValue1 = Bind<Vec::Sub<T2>, DivValue, DupOne>;
    using SelectRes = Bind<Vec::Select<uint8_t, T2, DIV_SEL_TENSOR_TENSOR_MODE>, ResMask, SubValue1, DivValue>;
    using CastOutHalf = Bind<Vec::Cast<half, T2, DIV_CAST_MODE_RINT>, SelectRes>;
    using CastOutInteger = Bind<CastOverFlow<T1, half, DIV_CAST_MODE_RINT>, CastOutHalf>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOutInteger>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct DivIntegerU8 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::Div<T2>, CastX1, CastX2>;
    using CastOut = Bind<Vec::Cast<T1, T2, DIV_CAST_MODE_NONE>, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct DivIntegerS32 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivValue = Bind<Vec::Div<T1>, InputX1, InputX2>;
    using MulValue = Bind<Vec::Mul<T1>, InputX2, DivValue>;
    using SubValue = Bind<Vec::Sub<T1>, InputX1, MulValue>;
    using ConstZero = MAKE_CONST(T1, 0);
    using DupZero = Bind<Vec::Duplicate<T1>, ConstZero>;
    using RemMask = Bind<Vec::Compare<uint8_t, T1, DIV_CMP_NE_MODE>, SubValue, DupZero>;
    using ConstFlag = MAKE_CONST(T1, DIV_B32_SIGN);
    using DupFlag = Bind<Vec::Duplicate<T1>, ConstFlag>;
    using AndX1 = Bind<Vec::And<T1>, InputX1, DupFlag>;
    using AndX2 = Bind<Vec::And<T1>, InputX2, DupFlag>;
    using SignMask = Bind<Vec::Compare<uint8_t, T1, DIV_CMP_NE_MODE>, AndX1, AndX2>;
    using ResMask = Bind<Vec::And<uint8_t>, RemMask, SignMask>;
    using ConstOne = MAKE_CONST(T1, 1);
    using DupOne = Bind<Vec::Duplicate<T1>, ConstOne>;
    using SubValue1 = Bind<Vec::Sub<T1>, DivValue, DupOne>;
    using SelectRes = Bind<Vec::Select<uint8_t, T1, DIV_SEL_TENSOR_TENSOR_MODE>, ResMask, SubValue1, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, SelectRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace DivOp

#endif // DIV_DAG_H
