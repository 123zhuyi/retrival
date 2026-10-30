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
 * \file rsqrt_grad_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_RSQRT_GRAD_DAG_H
#define CANN_CUSTOM_OPS_RSQRT_GRAD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace RsqrtGradDag {

#ifdef __CCE_AICORE__
constexpr static Reg::CastTrait cutsomCastTrait0 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};
constexpr static Reg::CastTrait castTraitFLOAT2INT32 = {
    Reg::RegLayout::UNKNOWN,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};
constexpr static Reg::CastTrait castTraitINT322FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};
constexpr static Reg::CastTrait castTraitB322B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr static Reg::CastTrait castTraitB162B8 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};
constexpr static Reg::CastTrait castTraitB82B16 = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
#endif

template <class T>
struct DivsCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline DivsCustom(LocalTensor<T>& dst, LocalTensor<T>& src, const T& scalar, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput1;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::RegTensor<T> vregInput2;

        Reg::MaskReg mask;
        __VEC_SCOPE__
        {
            Reg::Duplicate(vregInput2, scalar);
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                // OpCopyIn
                Reg::DataCopy(vregInput1, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                // OpCompute
                Reg::Div(vregOutput, vregInput1, vregInput2, mask);
                // OpCopyOut
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <class T>
struct RsqrtGradB8 : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline RsqrtGradB8(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T> vregInput1;
        Reg::RegTensor<T> vregInput2;
        Reg::RegTensor<T> vregOutput;

        Reg::RegTensor<int32_t> regTensor1, regTensor2;
        Reg::RegTensor<float> regTensor3, regTensor4;
        Reg::RegTensor<half> regTensor5, regTensor6;

        Reg::MaskReg mask;

        __VEC_SCOPE__
        {
            Reg::Duplicate(regTensor2, 255);
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<float>(count);
                // OpCopyIn
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK4_B8>(vregInput1,
                                                                 (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::Cast<half, T, castTraitB82B16>(regTensor5, vregInput1, mask);
                Reg::Cast<float, half, cutsomCastTrait0>(regTensor3, regTensor5, mask);

                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK4_B8>(vregInput2,
                                                                 (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
                Reg::Cast<half, T, castTraitB82B16>(regTensor6, vregInput2, mask);
                Reg::Cast<float, half, cutsomCastTrait0>(regTensor4, regTensor6, mask);

                // Compute z
                Reg::Mul<float>(regTensor4, regTensor4, regTensor3, mask);
                Reg::Mul<float>(regTensor3, regTensor3, regTensor3, mask);
                Reg::Muls<float>(regTensor4, regTensor4, (float)(-0.5), mask);
                Reg::Mul<float>(regTensor3, regTensor4, regTensor3, mask);

                // Compute cast to int8 with overflow wraparound
                Reg::Cast<int32_t, float, castTraitFLOAT2INT32>(regTensor1, regTensor3, mask);
                Reg::And<int32_t, Reg::MaskMergeMode::ZEROING>(regTensor1, regTensor1, regTensor2, mask);
                Reg::Cast<float, int32_t, castTraitINT322FLOAT>(regTensor3, regTensor1, mask);
                Reg::Adds<float>(regTensor3, regTensor3, (float)128, mask);
                // compute mod
                Reg::Muls<float>(regTensor4, regTensor3, (float)(0.00390625), mask);
                Reg::Truncate<float, RoundMode::CAST_TRUNC, Reg::MaskMergeMode::ZEROING>(regTensor4, regTensor4, mask);
                Reg::Muls<float>(regTensor4, regTensor4, (float)(256), mask);
                Reg::Sub<float>(regTensor3, regTensor3, regTensor4, mask);
                Reg::Adds<float>(regTensor3, regTensor3, (float)(-128), mask);
                // cast to int8
                Reg::Cast<half, float, castTraitB322B16>(regTensor5, regTensor3, mask);
                Reg::Cast<T, half, castTraitB162B8>(vregOutput, regTensor5, mask);
                // OpCopyOut
                Reg::DataCopy<T, Reg::StoreDist::DIST_PACK4_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput,
                                                                 mask);
            }
        }
#endif
    }
};
} // namespace RsqrtGradDag

template <typename T>
struct RsqrtGradDAG {
    using ConstValue = MAKE_CONST(float, -0.5);

    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;

    using OpMul0 = Bind<Vec::Mul<T>, OpCopyIn1, OpCopyIn0>;
    using OpMul1 = Bind<Vec::Mul<T>, OpCopyIn0, OpCopyIn0>;
    using OpMuls = Bind<Vec::Muls<T>, OpMul0, ConstValue>;
    using OpMul2 = Bind<Vec::Mul<T>, OpMuls, OpMul1>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpMul2>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct RsqrtGradInt8 {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;

    // do compute with custom op
    using OpRes = Bind<RsqrtGradDag::RsqrtGradB8<T>, OpCopyIn0, OpCopyIn1>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
}; // int8 compute dag

template <typename T>
struct RsqrtGradWithDiv {
    using ConstValue = MAKE_CONST(float, -2);

    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;

    // use div instead of muls
    using OpMul0 = Bind<Vec::Mul<T>, OpCopyIn1, OpCopyIn0>;
    using OpMul1 = Bind<Vec::Mul<T>, OpCopyIn0, OpCopyIn0>;
    using OpDivs = Bind<RsqrtGradDag::DivsCustom<T>, OpMul0, ConstValue>;
    using OpMul2 = Bind<Vec::Mul<T>, OpDivs, OpMul1>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpMul2>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
}; // int32 compute dag

#endif // CANN_CUSTOM_OPS_RSQRT_GRAD_DAG_H
