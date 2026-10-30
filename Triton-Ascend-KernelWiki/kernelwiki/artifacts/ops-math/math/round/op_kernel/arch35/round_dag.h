/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file round_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_ROUND_DAG_H
#define CANN_CUSTOM_OPS_ROUND_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace RoundDag {
using namespace Ops::Base;

constexpr int32_t CAST_MODE_NONE = 0;
constexpr int32_t CAST_MODE_RINT = 1;

constexpr int32_t COMPARE_MODE_LT = 0;
constexpr int32_t COMPARE_MODE_GT = 1;
constexpr int32_t COMPARE_MODE_EQ = 2;
constexpr int32_t COMPARE_MODE_GE = 4;
constexpr int32_t SELECT_MODE_VS = 1;

constexpr uint32_t UINT32_SIGN = 0x80000000;
constexpr uint16_t UINT16_SIGN = 0x8000;
constexpr int32_t ConstNegOne = -1;
constexpr int32_t ConstZero = 0;
constexpr int32_t ConstOne = 1;
constexpr int32_t ConstTwo = 2;

template <typename T>
struct RoundIntCustom : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline RoundIntCustom(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<T>& abs, const T& power,
                                     uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t vl = VECTOR_REG_WIDTH / sizeof(T);
        uint16_t loopNum = (count + vl - 1) / vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* absAddr = (__ubuf__ T*)abs.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregPow;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregDupZero;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregDupOne;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregDiv;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregMul;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregReUse;
        Reg::MaskReg mask;
        Reg::MaskReg cmpMaskReg0;
        Reg::MaskReg cmpMaskReg1;
        if constexpr (std::is_same_v<T, int32_t>) {
            __VEC_SCOPE__
            {
                Reg::Duplicate(vregDupZero, ConstZero);
                Reg::Duplicate(vregDupOne, ConstOne);
                Reg::Duplicate(vregPow, power);
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vl));
                    Reg::DataCopy(vregReUse, (__ubuf__ T*)(absAddr + loopIdx * vl));

                    Reg::Div(vregDiv, vregReUse, vregPow, mask);
                    Reg::Mul(vregMul, vregDiv, vregPow, mask);
                    Reg::Sub(vregReUse, vregReUse, vregMul, mask);
                    Reg::Muls(vregMul, vregReUse, ConstTwo, mask);

                    // + 1
                    Reg::CompareScalar<T, CMPMODE::EQ>(cmpMaskReg0, vregMul, power, mask);
                    Reg::And(vregReUse, vregDiv, vregDupOne, mask);
                    Reg::CompareScalar<T, CMPMODE::EQ>(cmpMaskReg1, vregReUse, ConstOne, mask);
                    Reg::MaskAnd(cmpMaskReg0, cmpMaskReg0, cmpMaskReg1, mask);
                    Reg::CompareScalar<T, CMPMODE::GT>(cmpMaskReg1, vregMul, power, mask);
                    Reg::MaskOr(cmpMaskReg0, cmpMaskReg0, cmpMaskReg1, mask);
                    Reg::Select(vregMul, vregDupOne, vregDupZero, cmpMaskReg0);
                    Reg::Add(vregMul, vregDiv, vregMul, mask);

                    Reg::CompareScalar<T, CMPMODE::LT>(cmpMaskReg0, vregInput, ConstZero, mask);
                    Reg::Neg(vregReUse, vregDupOne, mask);
                    Reg::Select(vregDiv, vregReUse, vregDupOne, cmpMaskReg0);
                    Reg::Mul(vregMul, vregMul, vregDiv, mask);
                    Reg::Mul(vregReUse, vregMul, vregPow, mask);

                    // OpCopyOut
                    Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vl), (Reg::RegTensor<T>&)vregReUse, mask);
                }
            }
        }
#endif
    }
};

template <class T>
struct RoundCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline RoundCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t vl = VECTOR_REG_WIDTH / sizeof(T);
        uint16_t loopNum = (count + vl - 1) / vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        if constexpr (std::is_same_v<T, float>) {
            Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vregOutInt;
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vl));

                    Reg::Truncate<T, RoundMode::CAST_RINT, Reg::MaskMergeMode::ZEROING>(vregOutput, vregInput, mask);
                    Reg::Duplicate(vregOutInt, UINT32_SIGN, mask);
                    Reg::And(vregOutInt, vregOutInt, (Reg::RegTensor<uint32_t>&)vregInput, mask);
                    Reg::Or(vregOutInt, vregOutInt, (Reg::RegTensor<uint32_t>&)vregOutput, mask);

                    // OpCopyOut
                    Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vl), (Reg::RegTensor<T>&)vregOutInt, mask);
                }
            }
        } else {
            Reg::RegTensor<uint16_t, Reg::RegTraitNumOne> vregOutInt;
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                    // OpCopyIn
                    Reg::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vl));

                    Reg::Truncate<T, RoundMode::CAST_RINT, Reg::MaskMergeMode::ZEROING>(vregOutput, vregInput, mask);
                    Reg::Duplicate(vregOutInt, UINT16_SIGN, mask);
                    Reg::And(vregOutInt, vregOutInt, (Reg::RegTensor<uint16_t>&)vregInput, mask);
                    Reg::Or(vregOutInt, vregOutInt, (Reg::RegTensor<uint16_t>&)vregOutput, mask);

                    // OpCopyOut
                    Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vl), (Reg::RegTensor<T>&)vregOutInt, mask);
                }
            }
        }
#endif
    }
};

template <typename U>
struct RoundInt {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyIn0>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct RoundIntConst {
    using OpDup = Bind<Vec::Duplicate<U>, Placeholder::Var<U, 0>>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpDup>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct RoundIntNegativeDecimalsInf {
    using ConstValueMin = MAKE_CONST(U, -2147483648);

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpAbs = Bind<Vec::Abs<U>, OpCopyIn0>;

    using OpRes = Bind<RoundIntCustom<U>, OpCopyIn0, OpAbs, Placeholder::Var<U, 0>>;

    using OpCompareGreaterThanNum = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, OpAbs, Placeholder::Var<U, 1>>;
    using OpCompareEqualMin = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_EQ>, OpCopyIn0, ConstValueMin>;
    using OpMaskOr = Bind<Vec::Or<uint8_t>, OpCompareGreaterThanNum, OpCompareEqualMin>;

    using OpSelect = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpMaskOr, ConstValueMin, OpRes>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpSelect>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct RoundIntNegativeDecimals {
    using ConstValueMin = MAKE_CONST(U, -2147483648);

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpAbs = Bind<Vec::Abs<U>, OpCopyIn0>;

    using OpRes = Bind<RoundIntCustom<U>, OpCopyIn0, OpAbs, Placeholder::Var<U, 0>>;

    using OpCompareEqualMin = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_EQ>, OpCopyIn0, ConstValueMin>;
    using OpSelect = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpCompareEqualMin, Placeholder::Var<U, 1>, OpRes>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpSelect>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct RoundIntNegativeDecimalsNine {
    using ConstValueNegOne = MAKE_CONST(U, -1);
    using ConstValueZero = MAKE_CONST(U, 0);
    using ConstValueOne = MAKE_CONST(U, 1);
    using ConstValueNeg20Y = MAKE_CONST(U, -2000000000);
    using ConstValue5Y = MAKE_CONST(U, 500000000);
    using ConstValue10Y = MAKE_CONST(U, 1000000000);
    using ConstValue15Y = MAKE_CONST(U, 1500000000);
    using ConstValue20Y = MAKE_CONST(U, 2000000000);
    using ConstValueMin = MAKE_CONST(U, -2147483648);

    using OpDupZero = Bind<Vec::Duplicate<U>, ConstValueZero>;
    using OpDupOne = Bind<Vec::Duplicate<U>, ConstValueOne>;

    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCompareLowerThanZero = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_LT>, OpCopyIn0, ConstValueZero>;
    using OpSelect = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpCompareLowerThanZero, ConstValueNegOne, OpDupOne>;
    using OpAbs = Bind<Vec::Abs<U>, OpCopyIn0>;

    // 0~500000000          = 0
    // 500000001~1499999999 = 1000000000
    // 1500000000~          = 2000000000
    using OpCompareGreaterThan5Y = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, OpAbs, ConstValue5Y>;
    using OpSelect2 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpCompareGreaterThan5Y, ConstValue10Y, OpDupZero>;
    using OpCompareGreaterEqual15Y = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GE>, OpAbs, ConstValue15Y>;
    using OpSelect3 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpCompareGreaterEqual15Y, ConstValue20Y, OpSelect2>;
    using OpMul = Bind<Vec::Mul<U>, OpSelect, OpSelect3>;

    using OpCompareEqualMin = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_EQ>, OpCopyIn0, ConstValueMin>;
    using OpSelect4 = Bind<Vec::Select<uint8_t, U, SELECT_MODE_VS>, OpCompareEqualMin, ConstValueNeg20Y, OpMul>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpSelect4>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U>
struct RoundZero {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpTruncate = Bind<RoundCustom<U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpTruncate>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundNan {
    using ConstValueZero = MAKE_CONST(T, 0.0);
    using OpDup = Bind<Vec::Duplicate<T>, ConstValueZero>;
    using OpDiv = Bind<Vec::Div<T>, OpDup, OpDup>;
    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpDiv>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundPositiveDecimals {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    using OpDup = Bind<Vec::Duplicate<T>, Placeholder::Var<T, 0>>;
    using OpMul = Bind<Vec::Mul<T>, OpCopyIn0Cast, OpDup>;
    using OpTruncate = Bind<RoundCustom<T>, OpMul>;
    using OpDiv = Bind<Vec::DivHighPrecision<T>, OpTruncate, OpDup>;

    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpDiv>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename U, typename T = float>
struct RoundNegativeDecimals {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;

    using OpDup = Bind<Vec::Duplicate<T>, Placeholder::Var<T, 0>>;
    using OpDiv = Bind<Vec::DivHighPrecision<T>, OpCopyIn0Cast, OpDup>;
    using OpTruncate = Bind<RoundCustom<T>, OpDiv>;
    using OpMul = Bind<Vec::Mul<T>, OpTruncate, OpDup>;

    using OpCopyOutCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpMul>;

    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCopyOutCast>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace RoundDag

#endif // CANN_CUSTOM_OPS_ROUND_DAG_H
