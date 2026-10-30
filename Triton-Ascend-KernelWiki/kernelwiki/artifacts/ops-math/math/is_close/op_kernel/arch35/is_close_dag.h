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
 * \file is_close_dag.h
 * \brief if features > 0 return gradients, else return *gradients.
 */

#ifndef IS_CLOSE_DAG_H
#define IS_CLOSE_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace IsCloseOp {
using namespace Ops::Base;
constexpr int COMPARE_MODE_LT = 0;
constexpr int COMPARE_MODE_E = 2;
constexpr int COMPARE_MODE_LE = 3;
constexpr int COMPARE_MODE_NE = 5;
constexpr int GREATER_SEL_MODE = 2;
constexpr float INF_CONST = INFINITY;

template <int equalNan>
struct NanEqualCompare : public Vec::ElemwiseQuaternaryOP<uint8_t, float, float, float, float> {
    __aicore__ inline NanEqualCompare(LocalTensor<uint8_t>& dst, LocalTensor<float>& src1, LocalTensor<float>& src2,
                                      float rtol, float atol, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ float* src1Addr = (__ubuf__ float*)src1.GetPhyAddr();
        __ubuf__ float* src2Addr = (__ubuf__ float*)src2.GetPhyAddr();
        __ubuf__ uint8_t* dstAddr = (__ubuf__ uint8_t*)dst.GetPhyAddr();

        Reg::RegTensor<float> x1Reg;
        Reg::RegTensor<float> x2Reg;
        Reg::RegTensor<float> absSubReg;
        Reg::RegTensor<float> x2AbsReg;
        Reg::RegTensor<float> rtolTmpReg;
        Reg::RegTensor<float> rtolReg;
        Reg::RegTensor<uint8_t> resultReg;
        Reg::RegTensor<float> regTensorInfinity;
        Reg::RegTensor<uint8_t> regTensorOne;
        Reg::RegTensor<uint8_t> regTensorZero;
        Reg::RegTensor<uint32_t> reguint32;
        Reg::RegTensor<uint16_t> reguint16;
        Reg::RegTensor<uint8_t> reguint8;
        // mask最多8个
        Reg::MaskReg mask;
        // 计算结果1、2使用
        Reg::MaskReg equalCmpMask; // 计算结果1,最后处理
        Reg::MaskReg tmpMask1;
        Reg::MaskReg tmpMask2;
        Reg::MaskReg bothNanMask; // 计算结果2
        // 计算结果3、4使用
        Reg::MaskReg funcCmpMask; // 计算结果3
        Reg::MaskReg finiteMask;  // 计算结果4
        // 最后处理
        Reg::MaskReg resultMask;

        if constexpr (equalNan == 1) {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    Reg::Duplicate(regTensorOne, (uint8_t)1.0, mask);
                    Reg::Duplicate(regTensorZero, (uint8_t)0.0, mask);
                    Reg::DataCopy(x1Reg, (__ubuf__ float*)(src1Addr + loopIdx * vlSize));
                    Reg::DataCopy(x2Reg, (__ubuf__ float*)(src2Addr + loopIdx * vlSize));

                    // 计算结果1——equalCmpMask
                    Reg::Compare<float, CMPMODE::EQ>(equalCmpMask, x1Reg, x2Reg, mask);

                    // x1x2和自己对比，输出False的位置，为Nan的位置
                    Reg::Compare<float, CMPMODE::NE>(tmpMask1, x1Reg, x1Reg, mask);
                    Reg::Compare<float, CMPMODE::NE>(tmpMask2, x2Reg, x2Reg, mask);
                    // MaskAnd取x1x2都是Nan的位置---------------
                    Reg::MaskAnd(bothNanMask, tmpMask1, tmpMask2, mask);
                    // MaskOr取计算结果1、4都为True的位置，结果存储到tmpMask2
                    Reg::MaskOr(tmpMask2, bothNanMask, equalCmpMask, mask);

                    // 计算结果3
                    Reg::FusedAbsSub(absSubReg, x1Reg, x2Reg, mask); // 中间结果
                    Reg::Abs(x2AbsReg, x2Reg, mask);
                    Reg::Muls(rtolTmpReg, x2AbsReg, rtol, mask);
                    Reg::Adds(rtolReg, rtolTmpReg, atol, mask);
                    Reg::Compare<float, CMPMODE::LE>(funcCmpMask, absSubReg, rtolReg, mask);

                    // 计算结果4
                    Reg::Duplicate(regTensorInfinity, (float)INF_CONST, mask);
                    Reg::Compare<float, CMPMODE::LT>(finiteMask, x2AbsReg, regTensorInfinity, mask);

                    // 其他处理
                    // 符合公式且有限
                    Reg::MaskAnd(tmpMask1, funcCmpMask, finiteMask, mask);
                    // 符合上一条或者完全一致
                    Reg::MaskOr(resultMask, tmpMask2, tmpMask1, mask);

                    Reg::Select(resultReg, regTensorOne, regTensorZero, resultMask);
                    Reg::Pack(reguint16, (AscendC::Reg::RegTensor<uint32_t>&)resultReg);
                    Reg::Pack(reguint8, reguint16);

                    AscendC::Reg::MaskPack<AscendC::Reg::HighLowPart::LOWEST>(tmpMask1, mask);
                    AscendC::Reg::MaskPack<AscendC::Reg::HighLowPart::LOWEST>(mask, tmpMask1);
                    // OpCopyOut
                    Reg::DataCopy((__ubuf__ uint8_t*)(dstAddr + loopIdx * vlSize), reguint8, mask);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                    mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                    Reg::Duplicate(regTensorOne, (uint8_t)1.0f, mask);
                    Reg::Duplicate(regTensorZero, (uint8_t)0.0f, mask);
                    Reg::DataCopy(x1Reg, (__ubuf__ float*)(src1Addr + loopIdx * vlSize));
                    Reg::DataCopy(x2Reg, (__ubuf__ float*)(src2Addr + loopIdx * vlSize));
                    // 计算结果1——equalCmpMask
                    Reg::Compare<float, CMPMODE::EQ>(equalCmpMask, x1Reg, x2Reg, mask);

                    // 计算结果3
                    Reg::FusedAbsSub(absSubReg, x1Reg, x2Reg, mask); // 中间结果
                    Reg::Abs(x2AbsReg, x2Reg, mask);
                    Reg::Muls(rtolTmpReg, x2AbsReg, rtol, mask);
                    Reg::Adds(rtolReg, rtolTmpReg, atol, mask);
                    Reg::Compare<float, CMPMODE::LE>(funcCmpMask, absSubReg, rtolReg, mask);

                    // 计算结果4
                    Reg::Duplicate(regTensorInfinity, (float)INF_CONST, mask);
                    Reg::Compare<float, CMPMODE::LT>(finiteMask, x2AbsReg, regTensorInfinity, mask);

                    // 其他处理
                    // 符合公式且有限
                    Reg::MaskAnd(tmpMask1, funcCmpMask, finiteMask, mask);
                    // 符合上一条或者完全一致
                    Reg::MaskOr(resultMask, equalCmpMask, tmpMask1, mask);

                    Reg::Select(resultReg, regTensorOne, regTensorZero, resultMask);
                    Reg::Pack(reguint16, (AscendC::Reg::RegTensor<uint32_t>&)resultReg);
                    Reg::Pack(reguint8, reguint16);
                    AscendC::Reg::MaskPack<AscendC::Reg::HighLowPart::LOWEST>(tmpMask1, mask);
                    AscendC::Reg::MaskPack<AscendC::Reg::HighLowPart::LOWEST>(mask, tmpMask1);

                    // OpCopyOut
                    Reg::DataCopy((__ubuf__ uint8_t*)(dstAddr + loopIdx * vlSize), reguint8, mask);
                }
            }
        }
#endif
    }
};

template <typename T>
struct IsCloseDag {
    // |x1 - x2| ≤ atol + rtol * |x2|
    // 数据搬入
    using InputX1T = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2T = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    // cast
    using InputX1 = Bind<Vec::Cast<float, T, 0>, InputX1T>;
    using InputX2 = Bind<Vec::Cast<float, T, 0>, InputX2T>;

    using OpResult = Bind<NanEqualCompare<0>, InputX1, InputX2, Placeholder::Var<float, 0>, Placeholder::Var<float, 1>>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct IsCloseEqualNanDag {
    // |x1 - x2| ≤ atol + rtol * |x2|
    // 数据搬入
    using InputX1T = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2T = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    // cast
    using InputX1 = Bind<Vec::Cast<float, T, 0>, InputX1T>;
    using InputX2 = Bind<Vec::Cast<float, T, 0>, InputX2T>;

    using OpResult = Bind<NanEqualCompare<1>, InputX1, InputX2, Placeholder::Var<float, 0>, Placeholder::Var<float, 1>>;

    using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, OpResult>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace IsCloseOp

#endif // IS_CLOSE_DAG_H
