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
 * \file abs_grad_dag.h
 * \brief z = dy * sign(y)
 */

#ifndef CANN_CUSTOM_OPS_ABS_GRAD_DAG_H
#define CANN_CUSTOM_OPS_ABS_GRAD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

namespace AbsGradOp {
using namespace AscendC;
using namespace Ops::Base;

// 常量定义 - float16
constexpr uint16_t FP16_POS_ONE = 0x3c00;
constexpr uint16_t FP16_NEG_ONE = 0xbc00;
constexpr uint16_t FP16_ZERO = 0x0000;

// 常量定义 - float32
constexpr uint32_t FP32_POS_ONE = 0x3f800000;
constexpr uint32_t FP32_NEG_ONE = 0xbf800000;
constexpr uint32_t FP32_ZERO = 0x00000000;

// 常量定义 - bfloat16
constexpr uint16_t BF16_POS_ONE = 0x3f80;
constexpr uint16_t BF16_NEG_ONE = 0xbf80;
constexpr uint16_t BF16_ZERO = 0x0000;

namespace AbsGradVf {

template <class T>
struct AbsGradCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline AbsGradCustom(LocalTensor<T>& dst, LocalTensor<T>& y, LocalTensor<T>& dy, uint32_t count)
    {
#ifdef __CCE_AICORE__
        // 寄存器定义
        Reg::RegTensor<T> vY;
        Reg::RegTensor<T> vDy;
        Reg::RegTensor<T> vSign;
        Reg::RegTensor<T> vResult;
        Reg::RegTensor<T> vZero;
        Reg::RegTensor<T> vOne;
        Reg::RegTensor<T> vNegOne;
        Reg::MaskReg preg;
        Reg::MaskReg pregZero;
        Reg::MaskReg pregPos;
        Reg::MaskReg pregNeg;

        uint32_t sreg = (uint32_t)count;
        constexpr uint32_t vflen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
        static constexpr uint32_t repeatStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(count, repeatStride));
        __ubuf__ T* yAddr = (__ubuf__ T*)(y.GetPhyAddr());
        __ubuf__ T* dyAddr = (__ubuf__ T*)(dy.GetPhyAddr());
        __ubuf__ T* dstAddr = (__ubuf__ T*)(dst.GetPhyAddr());

        __VEC_SCOPE__
        {
            // 加载常量
            if constexpr (std::is_same<T, float>::value) {
                Reg::Duplicate((AscendC::Reg::RegTensor<uint32_t>&)vZero, FP32_ZERO);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint32_t>&)vOne, FP32_POS_ONE);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint32_t>&)vNegOne, FP32_NEG_ONE);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint32_t>&)vSign, FP32_ZERO);
            } else if constexpr (std::is_same<T, half>::value) {
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vZero, FP16_ZERO);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vOne, FP16_POS_ONE);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vNegOne, FP16_NEG_ONE);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vSign, FP16_ZERO);
            } else {
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vZero, BF16_ZERO);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vOne, BF16_POS_ONE);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vNegOne, BF16_NEG_ONE);
                Reg::Duplicate((AscendC::Reg::RegTensor<uint16_t>&)vSign, BF16_ZERO);
            }

            for (uint16_t i = 0; i < repeatTime; i++) {
                preg = Reg::UpdateMask<T, Reg::RegTraitNumOne>(sreg);

                // 加载输入数据
                Reg::LoadAlign(vY, yAddr + i * vflen);
                Reg::LoadAlign(vDy, dyAddr + i * vflen);

                // 检测 y = 0(以及NAN情况)
                Reg::Compare<T, CMPMODE::EQ>(pregZero, vY, vZero, preg);

                Reg::Select(vSign, vZero, vY, pregZero);

                // 检测 y > 0
                Reg::Compare<T, CMPMODE::GT>(pregPos, vY, vZero, preg);

                // y > 0 → 1
                Reg::Select(vSign, vOne, vSign, pregPos);

                // 检测 y < 0
                Reg::Compare<T, CMPMODE::LT>(pregNeg, vY, vZero, preg);

                // y < 0 → -1
                Reg::Select(vSign, vNegOne, vSign, pregNeg);

                // 最终结果 z = sign(y) * dy
                Reg::Mul(vResult, vSign, vDy, preg);

                // 存储结果
                Reg::StoreAlign(dstAddr + i * vflen, vResult, preg);
            }
        }
#endif
    }
};

} // namespace AbsGradVf

template <typename T>
struct AbsGradDag {
    using OpCopyInY = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyInDy = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpResult = Bind<AbsGradVf::AbsGradCustom<T>, OpCopyInY, OpCopyInDy>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace AbsGradOp
#endif // CANN_CUSTOM_OPS_ABS_GRAD_DAG_H
