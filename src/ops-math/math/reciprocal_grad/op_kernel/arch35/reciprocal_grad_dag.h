/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reciprocal_grad_dag.h
 * \brief ReciprocalGrad 算子 DAG 计算图定义（atvoss 框架 - Elewise 模式）
 *
 * 计算公式: z = -y * y * dy
 * 优化处理: dy=0时直接返回0，避免产生-0和NaN
 *
 * VF实现: Reg 寄存器级运算，__VEC_SCOPE__ 内完成全部计算
 * - ReciprocalGradVF<T>: Muls(-1) → Mul(y) → Mul(dy) → CompareScalar(EQ) → Select(0)
 *
 * DAG数据流:
 * y  (GM) -> CopyIn (In0) -> Cast(fp16→fp32) ──┐
 *                                              → VF -> Cast(RINT, fp32→fp16) -> CopyOut -> z (GM)
 * dy (GM) -> CopyIn (In1) -> Cast(fp16→fp32) ──┘
 */

#ifndef RECIPROCAL_GRAD_DAG_H
#define RECIPROCAL_GRAD_DAG_H

#ifndef __CCE_AICORE__
#ifndef __aicore__
#define __aicore__
#endif
#endif

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

using namespace Ops::Base;

namespace NsReciprocalGrad {

constexpr int CAST_MODE_NONE = 0;
constexpr int CAST_MODE_RINT = 1;

template <typename T>
struct ReciprocalGradVF : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline ReciprocalGradVF(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, uint32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr static uint32_t dtypeSize = sizeof(T);
        constexpr static uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, VL);
        __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumOne> yReg;
            AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumOne> dyReg;
            AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumOne> resultReg;
            AscendC::Reg::RegTensor<T, AscendC::Reg::RegTraitNumOne> zeroReg;
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::MaskReg dyZeroMask;

            AscendC::Reg::Duplicate(zeroReg, static_cast<T>(0));
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = AscendC::Reg::UpdateMask<T, AscendC::Reg::RegTraitNumOne>(count);
                AscendC::Reg::LoadAlign(yReg, src0Addr + loopIdx * VL);
                AscendC::Reg::LoadAlign(dyReg, src1Addr + loopIdx * VL);

                AscendC::Reg::Muls(resultReg, yReg, static_cast<T>(-1), mask);
                AscendC::Reg::Mul(resultReg, resultReg, yReg, mask);
                AscendC::Reg::Mul(resultReg, resultReg, dyReg, mask);

                AscendC::Reg::Compares<T, CMPMODE::EQ>(dyZeroMask, dyReg, static_cast<T>(0), mask);
                AscendC::Reg::Select<T>(resultReg, zeroReg, resultReg, dyZeroMask);

                AscendC::Reg::StoreAlign(dstAddr + loopIdx * VL, resultReg, mask);
            }
        }
#endif
    }
};

template <typename T>
struct ReciprocalGradCompute {
    using OpInputY = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpInputDy = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using CastOpInputY = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, OpInputY>;
    using CastOpInputDy = Bind<Vec::Cast<float, T, CAST_MODE_NONE>, OpInputDy>;
    using OpVF = Bind<ReciprocalGradVF<float>, CastOpInputY, CastOpInputDy>;
    using CastOpVF = Bind<Vec::Cast<T, float, CAST_MODE_RINT>, OpVF>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, CastOpVF>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <>
struct ReciprocalGradCompute<float> {
    using OpInputY = Bind<Vec::CopyIn<float>, Placeholder::In0<float>>;
    using OpInputDy = Bind<Vec::CopyIn<float>, Placeholder::In1<float>>;
    using OpVF = Bind<ReciprocalGradVF<float>, OpInputY, OpInputDy>;
    using OpCopyOut = Bind<Vec::CopyOut<float>, Placeholder::Out0<float>, OpVF>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace NsReciprocalGrad

#endif // RECIPROCAL_GRAD_DAG_H
