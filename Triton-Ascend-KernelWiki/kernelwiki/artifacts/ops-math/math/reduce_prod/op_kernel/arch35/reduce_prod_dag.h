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
 * \file reduce_prod_dag.h
 * \brief reduce prod dag
 */

#ifndef REDUCE_PROD_DAG_H
#define REDUCE_PROD_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace ReduceProd {
using namespace AscendC;
using namespace Ops::Base;

template <class R, class T>
struct CastInt : public Vec::ElemwiseUnaryOP<R, T> {
    __aicore__ inline CastInt(const LocalTensor<R>& dst, const LocalTensor<T>& src, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_B16 = VECTOR_LENGTH / sizeof(int16_t);
        __local_mem__ T* srcAddr = (__local_mem__ T*)src.GetPhyAddr();
        __local_mem__ R* dstAddr = (__local_mem__ R*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_B16);

        __VEC_SCOPE__
        {
            Reg::RegTensor<T> srcValue;
            Reg::MaskReg preg;
            uint32_t sregMask = count;
            for (uint16_t j = 0; j < loopTimes; j++) {
                preg = Reg::UpdateMask<uint16_t>(sregMask);
                Reg::DataCopy<T, Reg::LoadDist::DIST_NORM>(srcValue, srcAddr + VL_B16 * j);
                Reg::DataCopy<R, Reg::StoreDist::DIST_PACK_B16>(dstAddr + VL_B16 * j, (Reg::RegTensor<R>&)srcValue,
                                                                preg);
            }
        }
#endif
    }
};

template <typename T, typename PromteT>
struct ReduceProdDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceProdOp<PromteT>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct ReduceProdI8Dag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceProdOp<PromteT>, Cast0>;
    using Cast1 = Bind<CastInt<T, PromteT>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ReduceProd

#endif
