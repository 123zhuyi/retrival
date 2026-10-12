/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file complex_abs_dag_complex.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_COMPLEX_ABS_DAG_COMPLEX_H
#define CANN_CUSTOM_OPS_COMPLEX_ABS_DAG_COMPLEX_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

namespace ComplexAbsOp {
using namespace AscendC;
using namespace Ops::Base;

namespace ComplexAbsVf {

template <class T, class U>
struct ComplexAbsCustom : public Vec::ElemwiseUnaryOP<T, U> {
    __aicore__ inline ComplexAbsCustom(LocalTensor<T>& dst, LocalTensor<U>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__

        // 常量定义 - float精度
        constexpr uint32_t MIN_NORMAL_FLOAT = 0x00800000u;
        constexpr float SCALE_UP = 16777216.0f;
        constexpr float SCALE_DOWN = 5.9604644775390625e-8f;
        constexpr int16_t SHR_NUM_FOR_FP32 = 23;
        constexpr uint32_t EXP_BIAS = 0x3F800000u; // 1.0的浮点表示(指数127，尾数0)

        using calcTypeInt = typename std::conditional<AscendC::Std::IsSame<T, half>::value, uint16_t, uint32_t>::type;

        // 寄存器定义
        Reg::RegTensor<T> vSrcReg0;
        Reg::RegTensor<T> vSrcReg1;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vA;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vB;
        Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vA0;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vA1;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vB1;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vAS;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vBS;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vA1Sq;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vB1Sq;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vSumSq;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vS;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vResult;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vResultS;

        Reg::RegTensor<float, Reg::RegTraitNumOne> vAbsSrcReg0;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vAbsSrcReg1;

        Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vMinNormalFloat;
        Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vExpBias;
        Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vZero;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vSCALE_UP;
        Reg::RegTensor<float, Reg::RegTraitNumOne> vSCALE_DOWN;

        uint32_t sreg = (uint32_t)count;
        Reg::MaskReg preg;
        Reg::MaskReg preg2;
        Reg::MaskReg preg3;
        Reg::MaskReg preg4;
        constexpr uint32_t vflen = AscendC::VECTOR_REG_WIDTH / sizeof(float);
        constexpr uint32_t ub = AscendC::VECTOR_REG_WIDTH / AscendC::ONE_BLOCK_SIZE;

        static constexpr uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, repeatStride));
        __ubuf__ T* srcAddr = (__ubuf__ T*)(src.GetPhyAddr());
        __ubuf__ T* dstAddr = (__ubuf__ T*)(dst.GetPhyAddr());
        __VEC_SCOPE__
        {
            // 加载常量
            Reg::Duplicate(vMinNormalFloat, MIN_NORMAL_FLOAT);
            Reg::Duplicate(vExpBias, EXP_BIAS);
            Reg::Duplicate(vSCALE_UP, SCALE_UP);
            Reg::Duplicate(vSCALE_DOWN, SCALE_DOWN);
            Reg::Duplicate(vZero, 0);

            static constexpr AscendC::Reg::CastTrait castTraitZero = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                RoundMode::UNKNOWN};
            static constexpr AscendC::Reg::CastTrait castTraitOne = {
                AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                RoundMode::UNKNOWN};
            static constexpr AscendC::Reg::CastTrait castTrait32to16 = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
                RoundMode::CAST_RINT};

            for (uint16_t i = 0; i < repeatTime; i++) {
                preg = Reg::UpdateMask<float, Reg::RegTraitNumOne>(sreg);
                if constexpr (std::is_same<U, complex32>::value) {
                    preg2 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
                    Reg::LoadAlign(vSrcReg0, srcAddr + i * vflen * 2);
                    Reg::Cast<float, T, castTraitZero>(vA, vSrcReg0, preg2);
                    Reg::Cast<float, T, castTraitOne>(vB, vSrcReg0, preg2);
                    Reg::Mul(vA1Sq, vA, vA, preg);
                    Reg::Mul(vB1Sq, vB, vB, preg);
                    Reg::Add(vSumSq, vA1Sq, vB1Sq, preg);
                    Reg::Sqrt(vS, vSumSq, preg);
                    Reg::Cast<T, float, castTrait32to16>(vResult, vS, preg);
                    Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dstAddr + i * vflen, vResult, preg);
                } else {
                    Reg::LoadAlign<T, Reg::LoadDist::DIST_DINTLV_B32>(vSrcReg0, vSrcReg1, srcAddr + i * vflen * 2);
                    Reg::Abs(vAbsSrcReg0, vSrcReg0, preg);
                    Reg::Abs(vAbsSrcReg1, vSrcReg1, preg);
                    Reg::Compare<float, CMPMODE::GT>(preg2, vAbsSrcReg1, vAbsSrcReg0, preg);
                    Reg::Select<float>(vA, vAbsSrcReg1, vAbsSrcReg0, preg2);
                    Reg::Select<float>(vB, vAbsSrcReg0, vAbsSrcReg1, preg2);

                    // 0的处理
                    Reg::Compare<uint32_t, CMPMODE::EQ>(preg4, (AscendC::Reg::RegTensor<uint32_t>&)vB, vZero, preg);

                    // subnormal处理
                    Reg::Compare<uint32_t, CMPMODE::LT>(preg3, (AscendC::Reg::RegTensor<uint32_t>&)vA, vMinNormalFloat,
                                                        preg);
                    Reg::Mul(vAS, vA, vSCALE_UP, preg);
                    Reg::Mul(vBS, vB, vSCALE_UP, preg);
                    Reg::Select(vA, vAS, vA, preg3);
                    Reg::Select(vB, vBS, vB, preg3);
                    // 提取A的指数因子（对vAForExp操作）
                    Reg::ShiftRights(vA0, (AscendC::Reg::RegTensor<uint32_t>&)vA, SHR_NUM_FOR_FP32, preg);
                    Reg::ShiftLefts(vA0, vA0, SHR_NUM_FOR_FP32, preg);
                    Reg::Sub((AscendC::Reg::RegTensor<uint32_t>&)vA1, (AscendC::Reg::RegTensor<uint32_t>&)vA, vA0,
                             preg);
                    Reg::Sub((AscendC::Reg::RegTensor<uint32_t>&)vB1, (AscendC::Reg::RegTensor<uint32_t>&)vB, vA0,
                             preg);

                    Reg::Add((AscendC::Reg::RegTensor<uint32_t>&)vA1, (AscendC::Reg::RegTensor<uint32_t>&)vA1, vExpBias,
                             preg);
                    Reg::Add((AscendC::Reg::RegTensor<uint32_t>&)vB1, (AscendC::Reg::RegTensor<uint32_t>&)vB1, vExpBias,
                             preg);

                    // 计算归化值
                    Reg::Mul(vA1Sq, vA1, vA1, preg);
                    Reg::Mul(vB1Sq, vB1, vB1, preg);
                    Reg::Add(vSumSq, vA1Sq, vB1Sq, preg);
                    Reg::Sqrt(vS, vSumSq, preg);

                    Reg::Mul(vResult, vS, (AscendC::Reg::RegTensor<float>&)vA0, preg);
                    Reg::Mul(vResultS, vResult, vSCALE_DOWN, preg);
                    Reg::Select(vResult, vResultS, vResult, preg3);

                    // 0的处理
                    Reg::Select(vResult, vA, vResult, preg4);

                    Reg::StoreAlign(dstAddr + i * vflen, vResult, preg);
                }
            }
        }
#endif
    }
};
} // namespace ComplexAbsVf

template <typename U, typename T = float>
struct ComplexAbsDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpResult = Bind<ComplexAbsVf::ComplexAbsCustom<T, U>, OpCopyIn0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ComplexAbsOp
#endif // CANN_CUSTOM_OPS_COMPLEX_ABS_DAG_H
