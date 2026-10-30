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
 * \file floor_mod_dag.h
 * \brief floor_mod dag
 */

#ifndef FLOOR_MOD_DAG_H
#define FLOOR_MOD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "op_kernel/math_util.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#endif

using namespace Ops::Base;
using namespace AscendC;

namespace FloorModOp {

constexpr int CAST_NONE_MODE = 0;

constexpr int FMOD_CMP_NE_MODE = 5;
constexpr int FMOD_SEL_TENSOR_TENSOR_MODE = 2;

constexpr uint32_t FMOD_B32_SIGN = 0X80000000;
constexpr uint64_t FMOD_B64_SIGN = 0x8000000000000000;

constexpr uint64_t FMOD_B64_MAX = 0Xffffffffffffffff;

#ifdef __CCE_AICORE__
constexpr static Reg::CastTrait castTrait1 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                              RoundMode::CAST_RINT};

#endif

template <class T>
struct FmodPostCompute : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline FmodPostCompute(LocalTensor<T>& dst, LocalTensor<T>& fmodRes, LocalTensor<T>& inputX2,
                                      const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* fmodResAddr = (__local_mem__ T*)fmodRes.GetPhyAddr();
        __local_mem__ T* inputX2Addr = (__local_mem__ T*)inputX2.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> zeroValue;
            Reg::RegTensor<T> fmodResValue;
            Reg::RegTensor<T> inputX2Value;
            Reg::RegTensor<T> addValue;
            Reg::RegTensor<T> resValue;

            Reg::RegTensor<uint32_t> signValue;
            Reg::RegTensor<uint32_t> fmodSignValue;
            Reg::RegTensor<uint32_t> inputX2signValue;

            Reg::MaskReg preg;
            Reg::MaskReg negValue;
            Reg::MaskReg signNegValue;
            Reg::MaskReg resMaskValue;
            uint32_t sregMask = count;

            Reg::Duplicate(zeroValue, T(0));
            Reg::Duplicate(signValue, FMOD_B32_SIGN);

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T>(sregMask);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(fmodResValue, fmodResAddr + VL_T * j);
                Reg::Compare<T, CMPMODE::NE>(negValue, fmodResValue, zeroValue, preg);

                Reg::And(fmodSignValue, (Reg::RegTensor<uint32_t>&)fmodResValue, signValue, preg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(inputX2Value, inputX2Addr + VL_T * j);
                Reg::Add(addValue, fmodResValue, inputX2Value, preg);
                Reg::And(inputX2signValue, (Reg::RegTensor<uint32_t>&)inputX2Value, signValue, preg);
                Reg::Compare<uint32_t, CMPMODE::NE>(signNegValue, fmodSignValue, inputX2signValue, preg);

                Reg::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                Reg::Select(resValue, addValue, fmodResValue, resMaskValue);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

template <class T1, class T2>
struct FmodCastFloatPostCompute : public Vec::ElemwiseBinaryOP<T1, T2, T2> {
    __aicore__ inline FmodCastFloatPostCompute(LocalTensor<T1>& dst, LocalTensor<T2>& fmodRes, LocalTensor<T2>& inputX2,
                                               const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T2);
        __local_mem__ T2* fmodResAddr = (__local_mem__ T2*)fmodRes.GetPhyAddr();
        __local_mem__ T2* inputX2Addr = (__local_mem__ T2*)inputX2.GetPhyAddr();
        __local_mem__ T1* dstAddr = (__local_mem__ T1*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T2> zeroValue;
            Reg::RegTensor<T2> fmodResValue;
            Reg::RegTensor<T2> inputX2Value;
            Reg::RegTensor<T2> addValue;
            Reg::RegTensor<T2> resValue;
            Reg::RegTensor<T1> resCastValue;

            Reg::RegTensor<uint32_t> signValue;
            Reg::RegTensor<uint32_t> fmodSignValue;
            Reg::RegTensor<uint32_t> inputX2signValue;

            Reg::MaskReg preg;
            Reg::MaskReg negValue;
            Reg::MaskReg signNegValue;
            Reg::MaskReg resMaskValue;
            uint32_t sregMask = count;

            Reg::Duplicate(zeroValue, T2(0));
            Reg::Duplicate(signValue, FMOD_B32_SIGN);

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T2>(sregMask);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(fmodResValue, fmodResAddr + VL_T * j);
                Reg::Compare<T2, CMPMODE::NE>(negValue, fmodResValue, zeroValue, preg);

                Reg::And(fmodSignValue, (Reg::RegTensor<uint32_t>&)fmodResValue, signValue, preg);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(inputX2Value, inputX2Addr + VL_T * j);
                Reg::Add(addValue, fmodResValue, inputX2Value, preg);
                Reg::And(inputX2signValue, (Reg::RegTensor<uint32_t>&)inputX2Value, signValue, preg);
                Reg::Compare<uint32_t, CMPMODE::NE>(signNegValue, fmodSignValue, inputX2signValue, preg);

                Reg::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                Reg::Select(resValue, addValue, fmodResValue, resMaskValue);
                Reg::Cast<T1, float, castTrait1>(resCastValue, resValue, preg);
                Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B32>(dstAddr + VL_T * j, resCastValue, preg);
            }
        }
#endif
    }
};

template <class T>
struct FmodIntPostCompute : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline FmodIntPostCompute(LocalTensor<T>& dst, LocalTensor<T>& input1, LocalTensor<T>& input2,
                                         LocalTensor<T>& div, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* input1Addr = (__local_mem__ T*)input1.GetPhyAddr();
        __local_mem__ T* input2Addr = (__local_mem__ T*)input2.GetPhyAddr();
        __local_mem__ T* divAddr = (__local_mem__ T*)div.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> zeroValue;
            Reg::RegTensor<T> defaultValue;
            Reg::RegTensor<T> signValue;
            Reg::RegTensor<T> input1Value;
            Reg::RegTensor<T> input2Value;
            Reg::RegTensor<T> divValue;
            Reg::RegTensor<T> mulValue;
            Reg::RegTensor<T> subValue;
            Reg::RegTensor<T> modValue;
            Reg::RegTensor<T> modSignValue;
            Reg::RegTensor<T> addValue;
            Reg::RegTensor<T> input2SignValue;
            Reg::RegTensor<T> resValue;

            Reg::MaskReg preg;
            Reg::MaskReg cmpValue;
            Reg::MaskReg negValue;
            Reg::MaskReg signNegValue;
            Reg::MaskReg resMaskValue;
            uint32_t sregMask = count;

            Reg::Duplicate(zeroValue, T(0));
            Reg::Duplicate(defaultValue, T(-1));
            Reg::Duplicate(signValue, FMOD_B32_SIGN);

            for (uint16_t j = 0; j < loopTimes; j++) {
                // handel -1
                preg = Reg::UpdateMask<T>(sregMask);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(input2Value, input2Addr + VL_T * j);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(divValue, divAddr + VL_T * j);
                Reg::Mul(mulValue, input2Value, divValue, preg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(input1Value, input1Addr + VL_T * j);
                Reg::Sub(subValue, input1Value, mulValue, preg);
                Reg::Compare<T, CMPMODE::NE>(cmpValue, input2Value, zeroValue, preg);
                Reg::Select(modValue, subValue, defaultValue, cmpValue);

                // post handel
                Reg::Add(addValue, modValue, input2Value, preg);
                Reg::Compare<T, CMPMODE::NE>(negValue, modValue, zeroValue, preg);
                Reg::And(input2SignValue, input2Value, signValue, preg);
                Reg::And(modSignValue, modValue, signValue, preg);
                Reg::Compare<T, CMPMODE::NE>(signNegValue, modSignValue, input2SignValue, preg);
                Reg::MaskAnd(resMaskValue, signNegValue, negValue, preg);
                Reg::Select(resValue, addValue, modValue, resMaskValue);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void FloorModInt_1(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2,
                                                                    int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        const auto rem = src1[index] % src2[index];
        bool signs_differ = ((rem < 0) != (src2[index] < 0));
        if (signs_differ && (rem != 0)) {
            dst[index] = rem + src2[index];
        } else {
            dst[index] = rem;
        }
    }
}
#endif

template <class T>
struct FloorModInt : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline FloorModInt(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        asc_vf_call<FloorModInt_1<T>>(dim3(1024), dst_1, src1_1, src2_1, count);
#endif
    }
};

template <typename T>
struct FloorModFloatWithCastOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;
    using FmodRes = Bind<Vec::FmodHighPrecision<float>, OpCastX1, OpCastX2>;
    using Output = Bind<FmodCastFloatPostCompute<T, float>, FmodRes, OpCastX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct FloorModFloatOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<Vec::FmodHighPrecision<T>, OpInputX1, OpInputX2>;
    using Output = Bind<FmodPostCompute<T>, FmodRes, OpInputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct FloorModInt32Op {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<FloorModInt<T>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FmodRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct FloorModInt64Op {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<FloorModInt<T>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FmodRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace FloorModOp

#endif // FLOOR_MOD_DAG_H
