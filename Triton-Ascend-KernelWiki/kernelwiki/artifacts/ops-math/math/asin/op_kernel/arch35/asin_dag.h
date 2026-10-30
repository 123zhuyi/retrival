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
 * \file asin_dag.h
 * \brief Asin dag
 */

#ifndef ASIN_DAG_H
#define ASIN_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#endif

namespace AsinDag {
using namespace Ops::Base;

constexpr int CastModeToFp32 = 0;
constexpr int CastModeToBf16 = 1;

constexpr uint32_t THREAD_NUM = 1024;

constexpr float NUM_ONE = 1.0f;
constexpr float NEG_ONE = -1.0f;

#ifdef __CCE_AICORE__
using namespace AscendC;

#endif // __CCE_AICORE__

#ifdef __CCE_AICORE__
template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void AsinSimtCompute(__ubuf__ T* x, __ubuf__ T* y, const int64_t totalNum)
{
 	for(int64_t i = threadIdx.x; i < totalNum; i += blockDim.x){
 	    y[i] = asinf(x[i]);
 	}
}
#endif

template <class T>
struct AsinCustom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline AsinCustom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        asc_vf_call<AsinSimtCompute<T>>(dim3(THREAD_NUM),srcAddr,dstAddr,count);
#endif
    }
};

template <typename T>
struct AsinOpDirect {
    using InputX = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Y = Bind<AsinDag::AsinCustom<T>, InputX>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Y>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

template <typename T>
struct AsinOpWithCast {
    using OpCopyIn = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0 = Bind<Vec::Cast<float, T, AsinDag::CastModeToFp32>, OpCopyIn>;
    using OpAsin = Bind<AsinDag::AsinCustom<float>, Cast0>;
    using Cast1 = Bind<Vec::Cast<T, float, AsinDag::CastModeToBf16>, OpAsin>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace AsinDag
#endif // ASIN_DAG_H
