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
 * \file reduce_all_dag.h
 * \brief reduce all dag
 */

#ifndef REDUCE_ALL_DAG_H
#define REDUCE_ALL_DAG_H

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"
#ifdef __CCE_AICORE__
#include "op_kernel/math_util.h"
#endif

namespace ReduceAll {
using namespace Ops::Base;
using OutDtype = uint8_t;

template <typename T, typename R>
struct CastZeroOne : public Vec::ElemwiseUnaryOP<T, R> {
    __aicore__ inline CastZeroOne(const LocalTensor<T>& dst, const LocalTensor<R>& src, const uint32_t& count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t VECTOR_LENGTH = GetVRegSize();
        constexpr uint32_t VL_LENGTH = VECTOR_LENGTH / sizeof(T);
        __local_mem__ R* srcAddr = (__local_mem__ R*)src.GetPhyAddr();
        __local_mem__ T* dstAddr = (__local_mem__ T*)dst.GetPhyAddr();
        uint16_t loopTimes = CeilDiv(count, VL_LENGTH);
        uint32_t InSize = count;
        uint32_t OutSize = count;
        __VEC_SCOPE__
        {
            Reg::RegTensor<R> srcReg;
            Reg::RegTensor<T> dstReg;
            Reg::MaskReg cmpMask;
            Reg::MaskReg InMask;
            Reg::MaskReg OutMask;
            for (uint16_t j = 0; j < loopTimes; j++) {
                InMask = Reg::UpdateMask<R>(InSize);
                OutMask = Reg::UpdateMask<T>(OutSize);
                Reg::DataCopy<R, Reg::PostLiteral::POST_MODE_UPDATE>(srcReg, srcAddr, VL_LENGTH);
                Reg::Compares<R, CMPMODE::NE>(cmpMask, srcReg, 0.0f, InMask);
                Reg::Duplicate(dstReg, 0);
                Reg::Duplicate(dstReg, 1, cmpMask);
                Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, dstReg, VL_LENGTH, OutMask);
            }
        }
#endif
    }
};

template <typename T, typename PromteT>
struct ReduceAllDagFloat {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<CastZeroOne<PromteT, T>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceAllOp<PromteT>, Cast0>;
    using Cast1 = Bind<Vec::Cast<OutDtype, PromteT, 0>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<OutDtype>, Placeholder::Out0<OutDtype>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T, typename PromteT>
struct ReduceAllDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
    using ReduceOp0 = Bind<Vec::ReduceAllOp<PromteT>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ReduceAll

#endif
