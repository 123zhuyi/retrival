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
 * \file coalesce_sparse.cpp
 * \brief
 */
#if __CCE_AICORE__ == 310
#include "coalesce_sparse_simt.h"
#else
#include "coalesce_sparse.h"
#endif

extern "C" __global__ __aicore__ void coalesce_sparse(GM_ADDR unique_len, GM_ADDR unique_indices, GM_ADDR indices,
                                                      GM_ADDR values, GM_ADDR new_indices, GM_ADDR new_value,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const CoalesceSparseTilingData* __restrict tilingDevice = &tilingData;
#if __CCE_AICORE__ == 310 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT64 && ORIG_DTYPE_INDICES == DT_INT64
    if (TILING_KEY_IS(100)) {
        KernelCoalesceSparseSimt<int64_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        KernelCoalesceSparseSimt<int64_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        KernelCoalesceSparseSimt<int64_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#elif __CCE_AICORE__ == 310 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT64 && ORIG_DTYPE_INDICES == DT_INT32
    if (TILING_KEY_IS(103)) {
        KernelCoalesceSparseSimt<int64_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(104)) {
        KernelCoalesceSparseSimt<int64_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(105)) {
        KernelCoalesceSparseSimt<int64_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#elif __CCE_AICORE__ == 310 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT32 && ORIG_DTYPE_INDICES == DT_INT64
    if (TILING_KEY_IS(106)) {
        KernelCoalesceSparseSimt<int32_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(107)) {
        KernelCoalesceSparseSimt<int32_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(108)) {
        KernelCoalesceSparseSimt<int32_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#elif __CCE_AICORE__ == 310 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT32 && ORIG_DTYPE_INDICES == DT_INT32
    if (TILING_KEY_IS(109)) {
        KernelCoalesceSparseSimt<int32_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(110)) {
        KernelCoalesceSparseSimt<int32_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(111)) {
        KernelCoalesceSparseSimt<int32_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#endif

#if __CCE_AICORE__ == 220 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT64 && ORIG_DTYPE_INDICES == DT_INT64
    if (TILING_KEY_IS(0)) {
        KernelCoalesceSparse<int64_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelCoalesceSparse<int64_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelCoalesceSparse<int64_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#elif __CCE_AICORE__ == 220 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT64 && ORIG_DTYPE_INDICES == DT_INT32
    if (TILING_KEY_IS(3)) {
        KernelCoalesceSparse<int64_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelCoalesceSparse<int64_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelCoalesceSparse<int64_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#elif __CCE_AICORE__ == 220 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT32 && ORIG_DTYPE_INDICES == DT_INT64
    if (TILING_KEY_IS(6)) {
        KernelCoalesceSparse<int32_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        KernelCoalesceSparse<int32_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        KernelCoalesceSparse<int32_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#elif __CCE_AICORE__ == 220 && ORIG_DTYPE_UNIQUE_INDICES == DT_INT32 && ORIG_DTYPE_INDICES == DT_INT32
    if (TILING_KEY_IS(9)) {
        KernelCoalesceSparse<int32_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        KernelCoalesceSparse<int32_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        KernelCoalesceSparse<int32_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
#endif
}
