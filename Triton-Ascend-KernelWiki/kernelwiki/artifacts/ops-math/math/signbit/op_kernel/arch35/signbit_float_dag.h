/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sign_float_dag.h
 * \brief sign_float_dag.h
 */

#ifndef SIGNBIT_FLOAT_DAG_H
#define SIGNBIT_FLOAT_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

using namespace Ops::Base;

namespace SignbitFloatOp {
constexpr int CAST_NONE_MODE = 0;
const int16_t STATE_BIT_SHF_VALUE = 31;
const int16_t DOUBLE_STATE_BIT_SHF_VALUE = 63;

template <class T>
struct FloatComputeCustom : public Vec::ElemwiseUnaryOP<uint8_t, T> {
    __aicore__ inline FloatComputeCustom(LocalTensor<uint8_t>& dst, LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_LENGTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ uint8_t* dstAddr = (__ubuf__ uint8_t*)dst.GetPhyAddr();

        AscendC::Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput1;
        AscendC::Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> vregOutput;
        AscendC::Reg::MaskReg mask;
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                // OpCopyIn
                AscendC::Reg::DataCopy(vregInput1, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));

                AscendC::Reg::ShiftRights<uint32_t, int16_t>(vregOutput, (Reg::RegTensor<uint32_t>&)vregInput1,
                                                             STATE_BIT_SHF_VALUE, mask);
                // OpCopyOut
                AscendC::Reg::DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(
                    dstAddr + loopIdx * vlSize, (Reg::RegTensor<uint8_t>&)vregOutput, mask);
            }
        }
#endif
    }
};

template <class T>
struct DoubleComputeCustom : public Vec::ElemwiseUnaryOP<uint8_t, T> {
    __aicore__ inline DoubleComputeCustom(LocalTensor<uint8_t>& dst, LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_LENGTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ uint8_t* dstAddr = (__ubuf__ uint8_t*)dst.GetPhyAddr();

        AscendC::Reg::RegTensor<uint64_t, Reg::RegTraitNumOne> vregOutput;
        AscendC::Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput1;
        AscendC::Reg::MaskReg tmpMask;
        AscendC::Reg::MaskReg mask;
        uint32_t countTmp = count;
        Reg::RegTensor<uint32_t, Reg::RegTraitNumOne> tmpReg;
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                // OpCopyIn
                AscendC::Reg::DataCopy(vregInput1, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));

                AscendC::Reg::ShiftRights<uint64_t, int16_t>(vregOutput, (Reg::RegTensor<uint64_t>&)vregInput1,
                                                             DOUBLE_STATE_BIT_SHF_VALUE, mask);
                Reg::Pack<uint32_t, uint64_t, Reg::HighLowPart::LOWEST>(tmpReg, vregOutput);
                Reg::Pack<Reg::HighLowPart::LOWEST>(tmpMask, mask);

                // OpCopyOut
                AscendC::Reg::DataCopy<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(
                    dstAddr + loopIdx * vlSize, (Reg::RegTensor<uint8_t>&)tmpReg, tmpMask);
            }
        }
#endif
    }
};
} // namespace SignbitFloatOp

template <class T>
struct SignbitFloatCompute {
    // 通过Compute构造计算图
    using Input0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using OpCastIn0 = Bind<Vec::Cast<float, T, SignbitFloatOp::CAST_NONE_MODE>, Input0>;

    using OpFloatCompute = Bind<SignbitFloatOp::FloatComputeCustom<float>, OpCastIn0>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, OpFloatCompute>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <class T>
struct SignbitDoubleCompute {
    // 通过Compute构造计算图
    using Input0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

    using OpDoubleCompute = Bind<SignbitFloatOp::DoubleComputeCustom<T>, Input0>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, OpDoubleCompute>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

#endif // SIGNBIT_FLOAT_DAG_H
