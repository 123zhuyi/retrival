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
 * \file gcd_dag.h
 * \brief gcd dag
 */

#ifndef GCD_DAG_H
#define GCD_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "simt_api/asc_simt.h"
#endif

using namespace Ops::Base;
using namespace AscendC;

namespace GcdOp {

union U {
    uint32_t i[2];
    uint64_t u;
};

#ifdef __CCE_AICORE__
template <typename T>
__simt_vf__ __aicore__
    LAUNCH_BOUND(512) inline void GcdVecInt64(__ubuf__ T* dst, __ubuf__ T* src1, __ubuf__ T* src2, int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        U a;
        U b;
        U c;
        a.u = static_cast<uint64_t>(src1[index]);
        b.u = static_cast<uint64_t>(src2[index]);

        uint64_t mask;
        mask = static_cast<uint64_t>(src1[index] >> 63);
        a.u = (a.u ^ mask) - mask;

        mask = static_cast<uint64_t>(src2[index] >> 63);
        b.u = (b.u ^ mask) - mask;

        if (a.u == 0) {
            dst[index] = b.u;
            continue;
        }
        if (b.u == 0) {
            dst[index] = a.u;
            continue;
        }

        uint8_t offset;

        offset = 0;
        c.u = a.u | b.u;
        if (c.i[0] != 0) {
            offset = __builtin_ffs(c.i[0]);
        } else if (c.i[1] != 0) {
            offset = __builtin_ffs(c.i[1]) + 32;
        }
        uint8_t shift = offset - 1;

        offset = 0;
        if (a.i[0] != 0) {
            offset = __builtin_ffs(a.i[0]);
        } else if (a.i[1] != 0) {
            offset = __builtin_ffs(a.i[1]) + 32;
        }
        a.u >>= offset - 1;

        offset = 0;
        if (b.i[0] != 0) {
            offset = __builtin_ffs(b.i[0]);
        } else if (b.i[1] != 0) {
            offset = __builtin_ffs(b.i[1]) + 32;
        }
        b.u >>= offset - 1;

        while (b.u != 0) {
            offset = 0;
            if (b.i[0] != 0) {
                offset = __builtin_ffs(b.i[0]);
            } else if (b.i[1] != 0) {
            offset = __builtin_ffs(b.i[1]) + 32;
            }
            b.u >>= offset - 1;
            if (a.u > b.u) {
                uint64_t temp = a.u;
                a.u = b.u;
                b.u = temp;
            }
            b.u = b.u - a.u;
        }
        dst[index] = a.u << shift;
    }
}
#endif

#ifdef __CCE_AICORE__
template <typename T1, typename T2>
__simt_vf__ __aicore__
    LAUNCH_BOUND(1024) inline void GcdVec(__ubuf__ T1* dst, __ubuf__ T1* src1, __ubuf__ T1* src2, uint8_t attr, int count)
{
    for (uint32_t index = static_cast<uint32_t>(threadIdx.x); index < count;
         index += static_cast<uint32_t>(blockDim.x)) {
        T2 a = static_cast<T2>(src1[index]);
        T2 b = static_cast<T2>(src2[index]);
        if constexpr (!IsSameType<T1, T2>::value) {
            T2 mask;
            mask = static_cast<T2>(src1[index] >> attr);
            a = (a ^ mask) - mask;

            mask = static_cast<T2>(src2[index] >> attr);
            b = (b ^ mask) - mask;
        }

        if (a == 0) {
            dst[index] = b;
            continue;
        }
        if (b == 0) {
            dst[index] = a;
            continue;
        }

        uint8_t shift = __builtin_ffs(a | b) - 1;
        a >>= __builtin_ffs(a) - 1;
        b >>= __builtin_ffs(b) - 1;
        while (b != 0) {
            b >>= __builtin_ffs(b) - 1;
            if (a > b) {
                T2 temp = a;
                a = b;
                b = temp;
            }
            b = b - a;
        }
        dst[index] = a << shift;
    }
}
#endif

template <class T>
struct GcdNode : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline GcdNode(LocalTensor<T>& dst, LocalTensor<T>& src1, LocalTensor<T>& src2, int count)
    {
#ifdef __CCE_AICORE__
        __ubuf__ T* dst_1 = (__ubuf__ T*)dst.GetPhyAddr();
        __ubuf__ T* src1_1 = (__ubuf__ T*)src1.GetPhyAddr();
        __ubuf__ T* src2_1 = (__ubuf__ T*)src2.GetPhyAddr();
        if constexpr (IsSameType<T, int64_t>::value) {
            asc_vf_call<GcdVecInt64<int64_t>>(dim3(512), dst_1, src1_1, src2_1, count);
        } else if constexpr (IsSameType<T, int32_t>::value) {
            asc_vf_call<GcdVec<int32_t, uint32_t>>(dim3(1024), dst_1, src1_1, src2_1, static_cast<uint8_t>(31), count);
        } else if constexpr (IsSameType<T, int16_t>::value) {
            asc_vf_call<GcdVec<int16_t, uint16_t>>(dim3(1024), dst_1, src1_1, src2_1, static_cast<uint8_t>(15), count);
        } else if constexpr (IsSameType<T, int8_t>::value) {
            asc_vf_call<GcdVec<int8_t, uint8_t>>(dim3(1024), dst_1, src1_1, src2_1, static_cast<uint8_t>(7), count);
        } else if constexpr (IsSameType<T, uint8_t>::value) {
            asc_vf_call<GcdVec<uint8_t, uint8_t>>(dim3(1024), dst_1, src1_1, src2_1, static_cast<uint8_t>(7), count);
        }
#endif
    }
};


template <typename T>
struct GcdCompute {
    using InputX1 = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2 = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;
    using GcdRes = Bind<GcdNode<T>, InputX1, InputX2>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, GcdRes>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace GcdOp

#endif // GCD_DAG_H
