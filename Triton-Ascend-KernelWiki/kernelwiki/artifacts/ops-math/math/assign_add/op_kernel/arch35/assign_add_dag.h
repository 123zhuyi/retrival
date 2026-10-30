/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file assign_add_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_ASSIGN_ADD_DAG_H
#define CANN_CUSTOM_OPS_ASSIGN_ADD_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

const int CAST_MODE_DEFAULT = 0;

namespace AssignAddDag {
using namespace Ops::Base;
template <class T>
struct AddCustom : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline AddCustom(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(T);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = CeilDivision(count, vl);
        uint32_t vlSize = vl;
        __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2Addr = (__ubuf__ T*)src2.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput1;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregInput2;
        Reg::RegTensor<T, Reg::RegTraitNumOne> vregOutput;
        Reg::MaskReg mask;
        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(count);
                // OpCopyIn
                Reg::DataCopy(vregInput1, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                Reg::DataCopy(vregInput2, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));

                Reg::Add(vregOutput, vregInput1, vregInput2, mask);
                // OpCopyOut
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};

template <typename U, typename T>
struct AssignAddDAG {
    // 通过Compute构造计算图
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn1 = Bind<Vec::CopyIn<T>, Placeholder::In1<T>>;
    using OpCopyIn1Cast = Bind<Vec::Cast<U, T, CAST_MODE_DEFAULT>, OpCopyIn1>;

    using OpAdd = Bind<AssignAddDag::AddCustom<U>, OpCopyIn0, OpCopyIn1Cast>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpAdd>;
    // 指定输出节点
    using Outputs = Elems<OpCopyOut>; // 设置输出
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace AssignAddDag

#endif // CANN_CUSTOM_OPS_ASSIGN_ADD_DAG_H
