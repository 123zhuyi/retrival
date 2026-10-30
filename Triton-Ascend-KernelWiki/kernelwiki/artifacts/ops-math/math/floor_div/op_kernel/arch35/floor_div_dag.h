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
 * \file floor_div_dag.h
 * \brief floor div dag
 */

#ifndef FLOOR_DIV_DAG_H
#define FLOOR_DIV_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#endif

using namespace Ops::Base;
using namespace AscendC;

namespace FloorDivOp {
constexpr int DIV_CAST_MODE_NONE = 0;
constexpr int DIV_CAST_MODE_RINT = 1;
constexpr int DIV_CAST_MODE_FLOOR = 2;
constexpr int DIV_CMP_NE_MODE = 5;
constexpr int DIV_SEL_TENSOR_TENSOR_MODE = 2;
constexpr int8_t SAT_POS = 60;
constexpr int16_t DIV_B16_SIGN = -32768;

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

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void FloorDivInt_1(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2,
                                                                    int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        bool signs_differ = ((src1[index] < 0) != (src2[index] < 0));
        if (signs_differ) {
            const auto quot = src1[index] / src2[index];
            const auto rem = src1[index] - src2[index] * quot;
            dst[index] = rem ? quot - 1 : quot;
        } else {
            dst[index] = src1[index] / src2[index];
        }
    }
}
#endif

template <class T>
struct FloorDivInt : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline FloorDivInt(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        asc_vf_call<FloorDivInt_1<T>>(dim3(1024), dst_1, src1_1, src2_1, count);
#endif
    }
};

template <typename T1>
struct FloorDivFloatWithoutCast {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivRes = Bind<Vec::DivHighPrecision<T1>, InputX1, InputX2>;
    using FloorRes = Bind<Vec::Truncate<T1, DIV_CAST_MODE_FLOOR>, DivRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, FloorRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct FloorDivFloatWithCast {
    // half and bfloat16
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivRes = Bind<Vec::DivHighPrecision<T2>, CastX1, CastX2>;
    using FloorRes = Bind<Vec::Truncate<T2, DIV_CAST_MODE_FLOOR>, DivRes>;
    using CastOut = Bind<Vec::Cast<T1, T2, DIV_CAST_MODE_RINT>, FloorRes>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, CastOut>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <class T1, class T2>
struct FloorDivIntS8PostCompute : public Vec::ElemwiseTernaryOP<T1, T2, T2, T2> {
    __aicore__ inline FloorDivIntS8PostCompute(LocalTensor<T1>& dst, LocalTensor<T2>& input1, LocalTensor<T2>& input2,
                                               LocalTensor<T2>& div, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        SetCtrlSpr<SAT_POS, SAT_POS>(0);
        constexpr uint32_t VL_T = VECTOR_REG_WIDTH / sizeof(T2);
        __local_mem__ T2* input1Addr = (__local_mem__ T2*)input1.GetPhyAddr();
        __local_mem__ T2* input2Addr = (__local_mem__ T2*)input2.GetPhyAddr();
        __local_mem__ T2* divAddr = (__local_mem__ T2*)div.GetPhyAddr();
        __local_mem__ T1* dstAddr = (__local_mem__ T1*)dst.GetPhyAddr();
        uint16_t loopTimes = (count + VL_T - 1) / VL_T;

        constexpr static Reg::CastTrait castTraitT2ToHalf = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                             Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        constexpr static Reg::CastTrait castTraitHalfToInt8 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                               Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        __VEC_SCOPE__
        {
            Reg::RegTensor<T2> zeroValue;
            Reg::RegTensor<T2> oneValue;
            Reg::RegTensor<T2> signValue;
            Reg::RegTensor<T2> input1Value;
            Reg::RegTensor<T2> input2Value;
            Reg::RegTensor<T2> divValue;
            Reg::RegTensor<T2> mulValue;
            Reg::RegTensor<T2> subValue;
            Reg::RegTensor<T2> resValue;
            Reg::RegTensor<T2> subValue1;
            Reg::RegTensor<half> halfValue;
            Reg::RegTensor<T1> int8Value;
            Reg::RegTensor<T2> input1SignValue;
            Reg::RegTensor<T2> input2SignValue;

            Reg::MaskReg preg;
            Reg::MaskReg negValue;
            Reg::MaskReg signNegValue;
            Reg::MaskReg resMaskValue;
            uint32_t sregMask = count;

            Reg::Duplicate(zeroValue, T2(0));
            Reg::Duplicate(oneValue, T2(1));
            Reg::Duplicate(signValue, T2(DIV_B16_SIGN));

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T2>(sregMask);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(input2Value, input2Addr + VL_T * j);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(divValue, divAddr + VL_T * j);
                Reg::Mul(mulValue, input2Value, divValue, preg);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(input1Value, input1Addr + VL_T * j);
                Reg::Sub(subValue, input1Value, mulValue, preg);
                Reg::Compare<T2, CMPMODE::NE>(negValue, subValue, zeroValue, preg);

                Reg::And(input1SignValue, input1Value, signValue, preg);
                Reg::And(input2SignValue, input2Value, signValue, preg);
                Reg::Compare<T2, CMPMODE::NE>(signNegValue, input1SignValue, input2SignValue, preg);

                Reg::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                Reg::Sub(subValue1, divValue, oneValue, preg);
                Reg::Select(resValue, subValue1, divValue, resMaskValue);

                Reg::Cast<half, T2, castTraitT2ToHalf>(halfValue, resValue, preg);
                Reg::Cast<T1, half, castTraitHalfToInt8>(int8Value, halfValue, preg);
                Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B16>(dstAddr + VL_T * j, int8Value, preg);
            }
        }
        SetCtrlSpr<SAT_POS, SAT_POS>(1);
#endif
    }
};

template <typename T1, typename T2>
struct FloorDivIntegerS8 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using CastX1 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX1>;
    using CastX2 = Bind<Vec::Cast<T2, T1, DIV_CAST_MODE_NONE>, InputX2>;
    using DivValue = Bind<Vec::Div<T2>, CastX1, CastX2>;
    using Output = Bind<FloorDivIntS8PostCompute<T1, T2>, CastX1, CastX2, DivValue>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct FloorDivIntegerU8 {
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
struct FloorDivIntegerS32 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivValue = Bind<FloorDivInt<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1>
struct FloorDivIntegerS64 {
    using InputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using InputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using DivValue = Bind<FloorDivInt<T1>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, DivValue>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace FloorDivOp

#endif // FLOOR_DIV_DAG_H
