/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_CUSTOM_OPS_ASSIGN_SUB_DAG_H
#define CANN_CUSTOM_OPS_ASSIGN_SUB_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace AssignSubDag {
using namespace Ops::Base;
template <class T>
struct SubCustom : public Ops::Base::Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline SubCustom(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, uint32_t count)
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

                Reg::Sub(vregOutput, vregInput1, vregInput2, mask);
                // OpCopyOut
                Reg::DataCopy((__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput, mask);
            }
        }
#endif
    }
};
} // namespace AssignSubDag

template <typename T>
struct AssignSubOp {
    // 通过Compute构造计算图
    // a-b
    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In0<T>>;
    using OpCopyIn1 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<T>, Ops::Base::Placeholder::In1<T>>;
    using OpSub = Ops::Base::Bind<AssignSubDag::SubCustom<T>, OpCopyIn0, OpCopyIn1>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<T>, Ops::Base::Placeholder::Out0<T>, OpSub>;
    // 指定输出节点
    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using OpDag = Ops::Base::DAGSch<Outputs>;
};

#endif // CANN_CUSTOM_OPS_ASSIGN_SUB_DAG_H
