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
 * \file addr_dag.h
 * \brief addr dag
 */

#ifndef ADDR_DAG_H
#define ADDR_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace AddrOp {
using namespace AscendC;
constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;
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

// α != 0; β == 0
template <typename T>
struct AddrWithoutBetaCommon {
    // 输入张量
    using InputVec1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputVec2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using Vec1xAlpha = Bind<Vec::Muls<T>, InputVec1, Placeholder::Var<T, 0>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<T>, Vec1xAlpha, InputVec2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Vec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β == 0; T == bf16 or fp16
template <typename T>
struct AddrWithoutBetaBf16Fp16 {
    // 输入张量
    using InputVec1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputVec2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using CastVec1 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputVec1>;
    using CastVec2 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputVec2>;

    using Vec1xAlpha = Bind<Vec::Muls<float>, CastVec1, Placeholder::Var<float, 0>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<float>, Vec1xAlpha, CastVec2>;

    using CastVec1xAlphaxVec2 = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, Vec1xAlphaxVec2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β == 0; T == int8_t
struct AddrWithoutBetaInt8 {
    // 输入张量
    using InputVec1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using InputVec2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using CastVec1 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputVec1>;
    using CastVec2 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputVec2>;

    using Vec1xAlpha = Bind<Vec::Muls<int32_t>, CastVec1, Placeholder::Var<int32_t, 0>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<int32_t>, CastVec2, Vec1xAlpha>;
    using Y = Bind<AndFF<int32_t>, Vec1xAlphaxVec2>;

    // cast成uint8_t 避免转int8_t溢出(饱和模式)
    using CastVec1xAlphaxVec2 = Bind<Vec::Cast<uint8_t, int32_t, CAST_MODE_NONE>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, CastVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β == 0; T == uint8_t
struct AddrWithoutBetaUint8 {
    // 输入张量
    using InputVec1 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In0<uint8_t>>;
    using InputVec2 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In1<uint8_t>>;
    using CastVec1 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputVec1>;
    using CastVec2 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputVec2>;

    using Vec1xAlpha = Bind<Vec::Muls<int32_t>, CastVec1, Placeholder::Var<int32_t, 0>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<uint32_t>, CastVec2, Vec1xAlpha>;
    using Y = Bind<AndFF<int32_t>, Vec1xAlphaxVec2>;

    using CastVec1xAlphaxVec2 = Bind<Vec::Cast<uint8_t, uint32_t, CAST_MODE_NONE>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, CastVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α == 0;
template <typename T>
struct AddrWithoutAlphaCommon {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;

    using SelfxBeta = Bind<Vec::Muls<T>, InputSelf, Placeholder::Var<T, 0>>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelfxBeta>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α == 0; T == bf16 or fp16
template <typename T>
struct AddrWithoutAlphaBf16Fp16 {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using CastSelf = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputSelf>;

    using SelfxBeta = Bind<Vec::Muls<float>, CastSelf, Placeholder::Var<float, 0>>;

    using CastSelfxBeta = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, SelfxBeta>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastSelfxBeta>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α == 0; T = int8_t
struct AddrWithoutAlphaInt8 {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using CastSelf = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputSelf>;

    using SelfxBeta = Bind<Vec::Muls<int32_t>, CastSelf, Placeholder::Var<int32_t, 0>>;
    using Y = Bind<AndFF<int32_t>, SelfxBeta>;

    // cast成uint8_t 避免转int8_t溢出(饱和模式)
    using CastSelfxBeta = Bind<Vec::Cast<uint8_t, int32_t, CAST_MODE_NONE>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, CastSelfxBeta>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α == 0; T = int8_t
struct AddrWithoutAlphaUint8 {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In0<uint8_t>>;
    using CastSelf = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputSelf>;

    using SelfxBeta = Bind<Vec::Muls<int32_t>, CastSelf, Placeholder::Var<int32_t, 0>>;
    using Y = Bind<AndFF<int32_t>, SelfxBeta>;

    using CastSelfxBeta = Bind<Vec::Cast<uint8_t, uint32_t, CAST_MODE_NONE>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, CastSelfxBeta>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β != 0;
template <typename T>
struct AddrWithBetaWithAlphaCommon {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputVec1 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputVec2 = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;

    using Vec1xAlpha = Bind<Vec::Muls<T>, InputVec1, Placeholder::Var<T, 1>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<T>, InputVec2, Vec1xAlpha>;

    using SelfxBeta = Bind<Vec::Muls<T>, InputSelf, Placeholder::Var<T, 0>>;

    using SelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Add<T>, SelfxBeta, Vec1xAlphaxVec2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, SelfxBetaAddVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β != 0; T == bf16 or fp16
template <typename T>
struct AddrWithBetaWithAlphaBf16Fp16 {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputVec1 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using InputVec2 = Bind<Vec::CopyInBrc<T>, Placeholder::In2<T>>;
    using CastSelf = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputSelf>;
    using CastVec1 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputVec1>;
    using CastVec2 = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, InputVec2>;

    using Vec1xAlpha = Bind<Vec::Muls<float>, CastVec1, Placeholder::Var<float, 1>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<float>, CastVec2, Vec1xAlpha>;

    using SelfxBeta = Bind<Vec::Muls<float>, CastSelf, Placeholder::Var<float, 0>>;

    using SelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Add<float>, SelfxBeta, Vec1xAlphaxVec2>;

    using CastSelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, SelfxBetaAddVec1xAlphaxVec2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastSelfxBetaAddVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β != 0; T == int8_t
struct AddrWithBetaWithAlphaInt8 {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In0<int8_t>>;
    using InputVec1 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In1<int8_t>>;
    using InputVec2 = Bind<Vec::CopyInBrc<int8_t>, Placeholder::In2<int8_t>>;
    using CastSelf = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputSelf>;
    using CastVec1 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputVec1>;
    using CastVec2 = Bind<Vec::Cast<int32_t, int8_t, CAST_MODE_NONE>, InputVec2>;

    using Vec1xAlpha = Bind<Vec::Muls<int32_t>, CastVec1, Placeholder::Var<int32_t, 1>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<int32_t>, CastVec2, Vec1xAlpha>;

    using SelfxBeta = Bind<Vec::Muls<int32_t>, CastSelf, Placeholder::Var<int32_t, 0>>;

    using SelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Add<int32_t>, SelfxBeta, Vec1xAlphaxVec2>;
    using Y = Bind<AndFF<int32_t>, SelfxBetaAddVec1xAlphaxVec2>;

    // cast成uint8_t 避免转int8_t溢出(饱和模式)
    using CastSelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Cast<uint8_t, int32_t, CAST_MODE_NONE>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<int8_t>, Placeholder::Out0<int8_t>, CastSelfxBetaAddVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};

// α != 0; β != 0; T == uint8_t
struct AddrWithBetaWithAlphaUint8 {
    // 输入张量
    using InputSelf = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In0<uint8_t>>;
    using InputVec1 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In1<uint8_t>>;
    using InputVec2 = Bind<Vec::CopyInBrc<uint8_t>, Placeholder::In2<uint8_t>>;
    using CastSelf = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputSelf>;
    using CastVec1 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputVec1>;
    using CastVec2 = Bind<Vec::Cast<uint32_t, uint8_t, CAST_MODE_NONE>, InputVec2>;

    using Vec1xAlpha = Bind<Vec::Muls<int32_t>, CastVec1, Placeholder::Var<int32_t, 1>>;

    using Vec1xAlphaxVec2 = Bind<Vec::Mul<uint32_t>, CastVec2, Vec1xAlpha>;

    using SelfxBeta = Bind<Vec::Muls<int32_t>, CastSelf, Placeholder::Var<int32_t, 0>>;

    using SelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Add<uint32_t>, SelfxBeta, Vec1xAlphaxVec2>;
    using Y = Bind<AndFF<int32_t>, SelfxBetaAddVec1xAlphaxVec2>;

    using CastSelfxBetaAddVec1xAlphaxVec2 = Bind<Vec::Cast<uint8_t, uint32_t, CAST_MODE_NONE>, Y>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, CastSelfxBetaAddVec1xAlphaxVec2>;
    using Output = Elems<OpCopyOut>;
    using OpDag = DAGSch<Output>;
};
} // namespace AddrOp

#endif // ADDR_DAG_H
