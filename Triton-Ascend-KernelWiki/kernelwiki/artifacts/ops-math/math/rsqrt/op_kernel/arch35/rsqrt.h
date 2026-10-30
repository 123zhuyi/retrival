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
 * \file rsqrt.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_RSQRT_H
#define CANN_CUSTOM_OPS_RSQRT_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace RsqrtDag {
using namespace Ops::Base;
constexpr int CastModeBf16ToFp32 = 0;
constexpr int CastModeFp32ToBf16 = 1;

template <class T>
struct RsqrtCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline RsqrtCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t VL = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = (count + VL - 1) / VL;
        uint32_t vlSize = VL;
        __VEC_SCOPE__
        {
            __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
            __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

            static constexpr Reg::DivSpecificMode mode = {Reg::MaskMergeMode::ZEROING, true};

            Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput;
            Reg::RegTensor<T, Reg::RegTraitNumOne> ones;
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregSqrt;
            Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
            Reg::MaskReg mask;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                // OpCopyIn0
                Reg::DataCopy(vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
                // compute rsqrt
                Reg::Duplicate(ones, (T)1.0, mask);
                Reg::Sqrt(vregSqrt, vregInput, mask);
                // high precision mode div
                Reg::Div<T, &mode>(vregOutput, ones, vregSqrt, mask);
                // OpCopyOut
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <typename T>
struct RsqrtOp {
    // 通过Compute构造计算图
    // y = 1 / sqrt(x)
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, CastModeBf16ToFp32>, OpCopyIn>;
    using OpRsqrt = Bind<RsqrtDag::RsqrtCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, CastModeFp32ToBf16>, OpRsqrt>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

}; // namespace RsqrtDag

#endif // CANN_CUSTOM_OPS_RSQRT_H
