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
 * \file dot_dag.h
 * \brief dot dag
 */

#ifndef DOT_DAG_H
#define DOT_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace Dot {
using namespace Ops::Base;

template <class R, class T>
struct CastInt : public Vec::ElemwiseUnaryOP<R, T> {
    __aicore__ inline CastInt(const LocalTensor<R>& dst, const LocalTensor<T>& src, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = 256U;
        constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);
        __local_mem__ T* srcAddr = (__local_mem__ T*)src.GetPhyAddr();
        __local_mem__ R* dstAddr = (__local_mem__ R*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_B32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> srcValue;
            Reg::MaskReg preg;
            uint32_t sregMask = count;
            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<uint32_t>(sregMask);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(srcValue, srcAddr + VL_B32 * j);
                Reg::DataCopy<R, Reg::StoreDist::DIST_PACK4_B32>(dstAddr + VL_B32 * j, (Reg::RegTensor<R>&)srcValue,
                                                                 preg);
            }
        }
#endif
    }
};
template <typename T, typename PromteT>
struct DotDagI8 {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using Mul0 = Bind<Vec::Mul<PromteT>, Cast0, Cast1>;
    using Reduce0 = Bind<Vec::ReduceSumOp<PromteT>, Mul0>;
    using Cast2 = Bind<CastInt<T, PromteT>, Reduce0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct DotDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using Cast1 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
    using Mul0 = Bind<Vec::Mul<PromteT>, Cast0, Cast1>;
    using Reduce0 = Bind<Vec::ReduceSumOp<PromteT>, Mul0>;
    using Cast2 = Bind<Vec::Cast<T, PromteT, 1>, Reduce0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast2>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace Dot

#endif
