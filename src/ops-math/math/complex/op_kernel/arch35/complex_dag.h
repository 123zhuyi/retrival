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
 * \file complex_dag.h
 * \brief complex operator BRC DAG definition for arch35 (ascend950)
 *
 * Complex operator: real(T) + imag(T) -> out(C) in interleaved format
 * Supports broadcast between real and imag inputs via CopyInBrc.
 * Uses AscendC::Interleave API for SIMD interleaving with double buffer pipeline.
 * T: float (float32) or half (float16)
 * C: complex64 (2*float32 packed) or complex32 (2*float16 packed), sizeof(C) = 2*sizeof(T)
 */
#ifndef COMPLEX_DAG_H_
#define COMPLEX_DAG_H_

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

#ifndef __CCE_AICORE__
struct complex64 {
    float real;
    float imag;
};
struct complex32 {
    Ops::Base::half real;
    Ops::Base::half imag;
};
#endif

namespace ComplexOp {
using namespace Ops::Base;

template <class C, class T>
struct ComplexMerge : public Vec::ElemwiseBinaryOP<C, T, T> {
    __aicore__ inline ComplexMerge(LocalTensor<C>& dst, LocalTensor<T>& real, LocalTensor<T>& imag, uint32_t count)
    {
#ifdef __CCE_AICORE__
        using namespace AscendC;
        constexpr uint32_t ALIGN_ELEMS = 32 / sizeof(T); // 32B的元素数量
        LocalTensor<T> dstT = dst.template ReinterpretCast<T>();

        uint32_t alignedCount = (count / ALIGN_ELEMS) * ALIGN_ELEMS;
        if (alignedCount > 0) {
            LocalTensor<T> dstHalf0 = dstT;
            LocalTensor<T> dstHalf1 = dstT[alignedCount];
            Interleave(dstHalf0, dstHalf1, real, imag, alignedCount);
        }

        // 非32B对齐的尾部数据
        for (uint32_t i = alignedCount; i < count; i++) {
            dstT.SetValue(i * 2, real.GetValue(i));
            dstT.SetValue(i * 2 + 1, imag.GetValue(i));
        }
#endif
    }
};

template <typename C, typename T>
struct ComplexBrcDag {
    using OpCopyInReal = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using OpCopyInImag = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using OpMerge = Bind<ComplexMerge<C, T>, OpCopyInReal, OpCopyInImag>;
    using OpCopyOut = Bind<Vec::CopyOut<C>, Placeholder::Out0<C>, OpMerge>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace ComplexOp
#endif // COMPLEX_DAG_H_
