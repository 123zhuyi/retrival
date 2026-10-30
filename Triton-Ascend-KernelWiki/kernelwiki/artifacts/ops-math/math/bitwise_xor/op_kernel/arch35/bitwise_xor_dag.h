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
 * \file bitwise_xor_dag.h
 * \brief bitwise xor dag
 */

#ifndef BITWISE_XOR_DAG_H
#define BITWISE_XOR_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "op_kernel/math_util.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

using namespace Ops::Base;
using namespace AscendC;

template <class T>
struct XorCompute : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline XorCompute(LocalTensor<T>& dst, LocalTensor<T>& inputX1, LocalTensor<T>& inputX2,
                                 const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
        __local_mem__ T* inputX1Addr = (__local_mem__ T*)inputX1.GetPhyAddr();
        __local_mem__ T* inputX2Addr = (__local_mem__ T*)inputX2.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_T);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> src1Value;
            Reg::RegTensor<T> src2Value;
            Reg::RegTensor<T> resValue;
            Reg::MaskReg preg;
            uint32_t sregMask = count;

            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<T>(sregMask);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src1Value, inputX1Addr + VL_T * j);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(src2Value, inputX2Addr + VL_T * j);

                Reg::Xor(resValue, src1Value, src2Value, preg);

                Reg::DataCopy<T, Reg::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
            }
        }
#endif
    }
};

namespace BitwiseXorOp {
using namespace AscendC;
template <typename T>
struct BitwiseXorCompute {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    using OpBitwiseXorRes = Bind<XorCompute<T>, InputX1, InputX2>;

    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpBitwiseXorRes>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace BitwiseXorOp
#endif // BITWISE_XOR_DAG_H
