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
 * \file tanh_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_TANH_DAG_H
#define CANN_CUSTOM_OPS_TANH_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

#ifdef __CCE_AICORE__
constexpr static AscendC::Reg::CastTrait castTrait0 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::Reg::CastTrait castTrait1 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                       AscendC::Reg::MaskMergeMode::ZEROING,
                                                       AscendC::RoundMode::CAST_RINT};
#endif
namespace TanhDag1 {
using namespace Ops::Base;
using namespace AscendC;

constexpr float FP32_ZERO_015 = 0.0157396831f;
constexpr float FP32_ZERO_NEG_052 = -0.0523039624f;
constexpr float FP32_ZERO_133 = 0.133152977f;
constexpr float FP32_ZERO_NEG_333 = -0.333327681f;
constexpr float FP32_TWO = 2.0f;
constexpr float FP32_ONE = 1.0f;
constexpr float FP32_ZERO_NEG_TWO = -2.0f;
constexpr float FP32_ZERO_6 = 0.60000002384185791016f;
constexpr float FP32_SAT_BOUND = 9.010913848876953125f;
constexpr uint32_t FP32_SIGN_MASK = 0x80000000;

#ifdef __CCE_AICORE__
struct TanhConst {
    Reg::RegTensor<float, Reg::RegTraitNumOne> vregValue1;
    Reg::RegTensor<float, Reg::RegTraitNumOne> vregValue2;
    Reg::RegTensor<float, Reg::RegTraitNumOne> vregOne;
    Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vregSignMask;
};
#endif

template <class T>
struct TanhCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline TanhCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t vl = VECTOR_REG_WIDTH / sizeof(float);
        uint16_t loopNum = CeilDivision(count, vl);
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        TanhConst constRegs;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInput;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput16;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput16;
        __VEC_SCOPE__
        {
            Reg::Duplicate(constRegs.vregValue1, FP32_ZERO_133);
            Reg::Duplicate(constRegs.vregValue2, FP32_ZERO_NEG_333);
            Reg::Duplicate(constRegs.vregOne, FP32_ONE);
            Reg::Duplicate(constRegs.vregSignMask, FP32_SIGN_MASK);
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                if constexpr (std::is_same_v<T, float>) {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vl));
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(vregInput16,
                                                                      (__ubuf__ T*)(srcAddr + loopIdx * vl));
                    Reg::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
                }
                TanhCoreCompute(constRegs, vregInput, vregOutput, mask);
                if constexpr (std::is_same_v<T, float>) {
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vl), vregOutput,
                                                                      mask);
                } else {
                    Reg::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vl),
                                                                      vregOutput16, mask);
                }
            }
        }
#endif
    }

#ifdef __CCE_AICORE__
private:
    __aicore__ inline void TanhCoreCompute(TanhConst& constRegs, Reg::RegTensor<float, Reg::RegTraitNumOne>& vregInput,
                                           Reg::RegTensor<float, Reg::RegTraitNumOne>& vregOutput, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputSqr;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputAbs;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vregInputMid;
        Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vregSign;
        Reg::MaskReg cmpMaskReg;
        Reg::MaskReg satMaskReg;
        Reg::Mul(vregInputSqr, vregInput, vregInput, mask);
        Reg::Muls(vregOutput, vregInputSqr, FP32_ZERO_015, mask);
        Reg::Adds(vregOutput, vregOutput, FP32_ZERO_NEG_052, mask);
        Reg::MulDstAdd(vregOutput, vregInputSqr, constRegs.vregValue1, mask);
        Reg::MulDstAdd(vregOutput, vregInputSqr, constRegs.vregValue2, mask);
        Reg::Mul(vregOutput, vregOutput, vregInputSqr, mask);
        Reg::MulDstAdd(vregOutput, vregInput, vregInput, mask);
        Reg::Abs(vregInputAbs, vregInput, mask);
        Reg::Muls(vregInputMid, vregInputAbs, FP32_TWO, mask);
        Reg::Exp(vregInputMid, vregInputMid, mask);
        Reg::Adds(vregInputMid, vregInputMid, FP32_ONE, mask);
        Reg::Div(vregInputMid, constRegs.vregOne, vregInputMid, mask);
        Reg::Muls(vregInputMid, vregInputMid, FP32_ZERO_NEG_TWO, mask);
        Reg::Adds(vregInputMid, vregInputMid, FP32_ONE, mask);
        Reg::Compares<float, CMPMODE::GE>(satMaskReg, vregInputAbs, FP32_SAT_BOUND, mask);
        Reg::Select(vregInputMid, constRegs.vregOne, vregInputMid, satMaskReg);
        Reg::And(vregSign, constRegs.vregSignMask, (Reg::RegTensor<uint32_t, Reg::RegTraitNumOne>&)vregInput, mask);
        Reg::Or((Reg::RegTensor<uint32_t, Reg::RegTraitNumOne>&)vregInputMid,
                (Reg::RegTensor<uint32_t, Reg::RegTraitNumOne>&)vregInputMid, vregSign, mask);
        Reg::Compares<float, CMPMODE::GE>(cmpMaskReg, vregInputAbs, FP32_ZERO_6, mask);
        Reg::Select(vregOutput, vregInputMid, vregOutput, cmpMaskReg);
    }
#endif // __CCE_AICORE__
};
} // namespace TanhDag1

template <typename U>
struct TanhDAG {
    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpResult1 = Ops::Base::Bind<TanhDag1::TanhCustom<U>, OpCopyIn0>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpResult1>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};

#endif // CANN_CUSTOM_OPS_TANH_DAG_H
