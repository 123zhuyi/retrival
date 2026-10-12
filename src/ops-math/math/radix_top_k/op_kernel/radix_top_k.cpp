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
 * \file radix_top_k.cpp
 * \brief radix top k op kernel 
 */
#include "kernel_operator.h"
#include "radix_top_k_struct.h"
#include "radix_top_k_tiling_key.h"
#include "radix_top_k_ub.h"
#include "radix_top_k_ws.h"
using namespace AscendC;

template <bool sorted, bool largest, bool isLargeShape>
__global__ __aicore__ void radix_top_k(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tpipe;
    REGISTER_TILING_DEFAULT(RadixTopKTilingData);
    GET_TILING_DATA(tilingData, tiling);
#if (defined(DTYPE_X))
    if constexpr (isLargeShape) {
        RadixTopK::RadixTopKWs<DTYPE_X, largest> op(tpipe, tilingData);
        op.Init(x, k, values, indices, workspace);
        op.Process();
    } else {
        RadixTopK::RadixTopKUb<DTYPE_X, largest> op(tpipe, tilingData);
        op.Init(x, k, values, indices, workspace);
        op.Process();
    }
#endif
}

// op_kernel UT 链接桩：仅提供 extern "C" 符号供链接，不参与实际执行逻辑。
#ifdef __CCE_UT_TEST__
extern "C" __global__ __aicore__ void radix_top_k(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices,
    GM_ADDR workspace, GM_ADDR tiling)
{
    radix_top_k<true, true, false>(x, k, values, indices, workspace, tiling);
}
#endif