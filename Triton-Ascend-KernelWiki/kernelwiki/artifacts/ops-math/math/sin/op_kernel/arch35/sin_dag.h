/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sin_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_SIN_DAG_H
#define CANN_CUSTOM_OPS_SIN_DAG_H
 
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

namespace SinOp
{
using namespace AscendC;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
constexpr uint32_t THREAD_NUM = 1024;

#ifdef __CCE_AICORE__
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SinSimtCompute(__ubuf__ T* x, __ubuf__ T* y, const int64_t totalNum)
{
    for(int64_t i = threadIdx.x; i < totalNum; i += blockDim.x){
        y[i] = Simt::Sin(x[i]);
    }
}
#endif

template<class T>
struct SinCustom : public Ops::Base::Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline SinCustom(LocalTensor<T> &dst, LocalTensor<T> &src, uint32_t count) {
#ifdef __CCE_AICORE__
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        Simt::VF_CALL<SinSimtCompute<T>>(Simt::Dim3(THREAD_NUM),srcAddr,dstAddr,count);
#endif
    }
};

template <typename U, typename T = float>
struct SinDAG {
    using OpCopyIn0 = Ops::Base::Bind<Ops::Base::Vec::CopyIn<U>, Ops::Base::Placeholder::In0<U>>;
    using OpCopyIn0Cast = Ops::Base::Bind<Ops::Base::Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpResult1 = Ops::Base::Bind<SinCustom<T>, OpCopyIn0Cast>;
    using OpResultCast = Ops::Base::Bind<Ops::Base::Vec::Cast<U, T, CAST_MODE_RINT>, OpResult1>;
    using OpCopyOut = Ops::Base::Bind<Ops::Base::Vec::CopyOut<U>, Ops::Base::Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Ops::Base::Elems<OpCopyOut>;
    using MemCfg = Ops::Base::MemOptCfg<Ops::Base::MemLevel::LEVEL_2>;
    using OpDag = Ops::Base::DAGSch<Outputs, void, MemCfg>;
};
}
#endif  // CANN_CUSTOM_OPS_SIN_DAG_H
