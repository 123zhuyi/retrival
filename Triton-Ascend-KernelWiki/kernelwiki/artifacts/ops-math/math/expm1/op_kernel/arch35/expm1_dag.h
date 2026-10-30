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
 * \file expm1_dag.h
 * \brief
 */

#ifndef OPS_MATH_EXPM1_DAG_H
#define OPS_MATH_EXPM1_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include <limits>

#ifdef __CCE_AICORE__
#include "simt_api/math_functions.h"
#include "simt_api/asc_simt.h"
#endif

namespace Expm1Op {
using namespace Ops::Base;
const int CAST_MODE_NONE = 0;
const int CAST_MODE_RINT = 1;
constexpr uint32_t THREAD_NUM = 1024;
constexpr float INV_LN2_APPROX = 1.4426950216293335f;
constexpr float LN2_HALF_APPROX = 0.4099999964237213f;
constexpr float LN2_APPROX = 0.693145751953125f;
constexpr float ONE_MINUS_LN2_APPROX = 0.000001428606765330187f;
constexpr float FLOAT_128 = 128.000000f;
constexpr float FLOAT_NEG_ONE = -1.00000000f;
constexpr float C5 = 0.008382412604987621f;
constexpr float C4 = 0.0013879507314413786f;
constexpr float C3 = 0.04166783019900322f;
constexpr float C2 = 0.1666639745235443f;
constexpr float C1 = 0.4999999403953552f;
constexpr float FLOAT_INF = std::numeric_limits<float>::infinity();
constexpr float FLOAT_NEG_25 = -25.0000000f;
constexpr float FLOAT_2 = 2.0000000f;

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void Expm1SimtCompute(__ubuf__ T* x, __ubuf__ T* y, const int64_t totalNum)
{
    for (int64_t i = threadIdx.x; i < totalNum; i += blockDim.x) {
        float f1 = x[i];
        float f0 = expm1f(f1);
        float f2 = f1 * INV_LN2_APPROX;
        float f3 = roundf(f2);
        float f4 = fabsf(f1);
        bool p1 = f4 < LN2_HALF_APPROX;
        float f5 = p1 ? 0.0f : f3;
        float f6 = -f5;
        float f7 = LN2_APPROX;
        float f8 = fmaf(f6, f7, f1);
        float f9 = ONE_MINUS_LN2_APPROX;
        float f10 = fmaf(f6, f9, f8);
        bool p2 = f5 == FLOAT_128;
        float f11 = f5 + FLOAT_NEG_ONE;
        float f12 = p2 ? f11 : f5;
        float f13 = C5;
        float f14 = C4;
        float f15 = fmaf(f14, f10, f13);
        float f16 = C3;
        float f17 = fmaf(f15, f10, f16);
        float f18 = C2;
        float f19 = fmaf(f17, f10, f18);
        float f20 = C1;
        float f21 = fmaf(f19, f10, f20);
        float f22 = f10 * f21;
        float f23 = fmaf(f22, f10, f10);
        float f24 = exp2f(f12);
        float f25 = f24 + FLOAT_NEG_ONE;
        float f26 = fmaf(f23, f24, f25);
        float f27 = f26 + f26;
        float f28 = p2 ? f27 : f26;
        bool p3 = f12 > FLOAT_128;
        float f29 = p3 ? FLOAT_INF : f28;
        bool p4 = f12 < FLOAT_NEG_25;
        float f30 = p4 ? FLOAT_NEG_ONE : f29;
        bool p5 = f1 == 0.0f;
        float f31 = f1 + f1;
        float f32 = p5 ? f31 : f30;
        y[i] = fabsf(x[i]) > FLOAT_2 ? f0 : f32;
    }
}
#endif

template <class T>
struct Expm1Custom : public Vec::ElemwiseUnaryOP<T, T> {
    __aicore__ inline Expm1Custom(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
        __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
        asc_vf_call<Expm1SimtCompute<T>>(dim3(THREAD_NUM), srcAddr, dstAddr, count);
#endif
    }
};

template <typename U, typename T = float>
struct Expm1DAG {
    using OpCopyIn0 = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyIn0Cast = Bind<Vec::Cast<T, U, CAST_MODE_NONE>, OpCopyIn0>;
    using OpResult1 = Bind<Expm1Custom<T>, OpCopyIn0Cast>;
    using OpResultCast = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResult1>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpResultCast>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace Expm1Op
#endif // OPS_MATH_EXPM1_DAG_H
