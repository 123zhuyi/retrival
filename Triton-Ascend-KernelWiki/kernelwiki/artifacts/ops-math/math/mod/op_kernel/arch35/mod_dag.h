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
 * \file mod_dag.h
 * \brief mod dag
 */

#ifndef MOD_DAG_H
#define MOD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/vec.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/math_util.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#endif

namespace ModOp {
using namespace Ops::Base;

constexpr int CAST_NONE_MODE = 0;
constexpr int CAST_RINT_MODE = 1;

template <class T>
struct ModIntPostCompute : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline ModIntPostCompute(const LocalTensor<T>& dst, const LocalTensor<T>& input1,
                                        const LocalTensor<T>& input2, const LocalTensor<T>& div, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = Ops::Base::GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* input1Addr = (__local_mem__ T*)input1.GetPhyAddr();
        __local_mem__ T* input2Addr = (__local_mem__ T*)input2.GetPhyAddr();
        __local_mem__ T* divAddr = (__local_mem__ T*)div.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = Ops::Base::CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> zeroValue;
            Reg::RegTensor<T> defaultValue;
            Reg::RegTensor<T> input1Value;
            Reg::RegTensor<T> input2Value;
            Reg::RegTensor<T> divValue;
            Reg::RegTensor<T> mulValue;
            Reg::RegTensor<T> subValue;
            Reg::RegTensor<T> resValue;
            Reg::MaskReg preg;
            Reg::MaskReg cmpValue;
            uint32_t sregMask = count;

            Reg::Duplicate(zeroValue, T(0));
            Reg::Duplicate(defaultValue, T(-1));

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T>(sregMask);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(input2Value, input2Addr + VL_T * j);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(divValue, divAddr + VL_T * j);
                Reg::Mul(mulValue, input2Value, divValue, preg);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(input1Value, input1Addr + VL_T * j);
                Reg::Sub(subValue, input1Value, mulValue, preg);
                Reg::Compare<T, CMPMODE::NE>(cmpValue, input2Value, zeroValue, preg);
                Reg::Select(resValue, subValue, defaultValue, cmpValue);
                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

template <typename T1, typename T2>
struct ModCastIntPostCompute : public Vec::ElemwiseTernaryOP<T1, T2, T2, T2> {
    __aicore__ inline ModCastIntPostCompute(const LocalTensor<T1>& dst, const LocalTensor<T2>& input1,
                                            const LocalTensor<T2>& input2, const LocalTensor<T2>& div,
                                            const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = Ops::Base::GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T2);
        __local_mem__ T2* input1Addr = (__local_mem__ T2*)input1.GetPhyAddr();
        __local_mem__ T2* input2Addr = (__local_mem__ T2*)input2.GetPhyAddr();
        __local_mem__ T2* divAddr = (__local_mem__ T2*)div.GetPhyAddr();
        __local_mem__ T1* dstAddr = (__local_mem__ T1*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T2> zeroValue;
            Reg::RegTensor<T2> defaultValue;
            Reg::RegTensor<T2> input1Value;
            Reg::RegTensor<T2> input2Value;
            Reg::RegTensor<T2> divValue;
            Reg::RegTensor<T2> mulValue;
            Reg::RegTensor<T2> subValue;
            Reg::RegTensor<T2> resValue;
            Reg::MaskReg preg;
            Reg::MaskReg cmpValue;
            uint32_t sregMask = count;

            Reg::Duplicate(zeroValue, T2(0));
            Reg::Duplicate(defaultValue, T2(-1));

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T2>(sregMask);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(input2Value, input2Addr + VL_T * j);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(divValue, divAddr + VL_T * j);
                Reg::Mul(mulValue, input2Value, divValue, preg);
                Reg::DataCopy<T2, Reg::LoadDist::DIST_NORM>(input1Value, input1Addr + VL_T * j);
                Reg::Sub(subValue, input1Value, mulValue, preg);
                Reg::Compare<T2, CMPMODE::NE>(cmpValue, input2Value, zeroValue, preg);
                Reg::Select(resValue, subValue, defaultValue, cmpValue);
                Reg::DataCopy<T1, Reg::StoreDist::DIST_PACK_B16>(dstAddr + VL_T * j, (Reg::RegTensor<T1>&)resValue,
                                                                 preg);
            }
        }
#endif
    }
};

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void ModIntSimt(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2,
                                                                 int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        const auto rem = src1[index] % src2[index];
        dst[index] = rem;
    }
}
#endif

template <class T>
struct ModInt : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline ModInt(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        asc_vf_call<ModIntSimt<T>>(dim3(1024), dst_1, src1_1, src2_1, count);
#endif
    }
};

template <typename T>
struct ModFloatWithCastOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpCastX1 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<float, T, CAST_NONE_MODE>, OpInputX2>;
    using FmodRes = Bind<Vec::FmodHighPrecision<float>, OpCastX1, OpCastX2>;
    using OpCastRes = Bind<Vec::Cast<T, float, CAST_RINT_MODE>, FmodRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct ModFloatOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<Vec::FmodHighPrecision<T>, OpInputX1, OpInputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FmodRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T1, typename T2>
struct ModIntWithCastOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T1>, Placeholder::In0<T1>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T1>, Placeholder::In1<T1>>;
    using OpCastX1 = Bind<Vec::Cast<T2, T1, CAST_NONE_MODE>, OpInputX1>;
    using OpCastX2 = Bind<Vec::Cast<T2, T1, CAST_NONE_MODE>, OpInputX2>;
    using DivRes = Bind<Vec::Div<T2>, OpCastX1, OpCastX2>;
    using Output = Bind<ModCastIntPostCompute<T1, T2>, OpCastX1, OpCastX2, DivRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T1>, Placeholder::Out0<T1>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct ModIntOp {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using DivRes = Bind<Vec::Div<T>, OpInputX1, OpInputX2>;
    using Output = Bind<ModIntPostCompute<T>, OpInputX1, OpInputX2, DivRes>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Output>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct ModInt64Op {
    using OpInputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpInputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using FmodRes = Bind<ModInt<T>, OpInputX1, OpInputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, FmodRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ModOp

#endif // MOD_DAG_H
