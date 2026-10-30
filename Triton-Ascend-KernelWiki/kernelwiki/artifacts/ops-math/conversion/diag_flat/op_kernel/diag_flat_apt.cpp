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
 * \file diag_flat_apt.cpp
 * \brief DiagFlat kernel entry for arch35 (DAV_3510, __NPU_ARCH__=3101)
 *
 * Single TilingKey 3501: SIMD+SIMT hybrid, covers all 13 dtypes via DTYPE_X.
 * (ref: diag_v2/op_kernel/diag_v2_apt.cpp same pattern)
 */

#include "arch35/diag_flat_tiling_key.h"
#include "arch35/diag_flat_simd.h"

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

template <int ARCH35_KEY>
__aicore__ inline void LaunchKernel(GM_ADDR x, GM_ADDR y,
                                     const DiagFlatArch35TilingData* td)
{
    // 8-byte types (complex64/float64/int64/uint64): alias through int64_t
    // because SIMT kernel cannot directly dereference __gm__ Complex<float>* or
    // __gm__ double*, nor static_cast<Complex<float>>(0) / static_cast<double>(0) in SIMT.
    if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        DiagFlatSimd<int64_t> op;
        op.Init(x, y, td);
        op.Process();
    } else {
        DiagFlatSimd<DTYPE_X> op;
        op.Init(x, y, td);
        op.Process();
    }
}

template <int ARCH35_KEY>
__global__ __aicore__ void diag_flat(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DiagFlatArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(DiagFlatArch35TilingData, tilingData, tiling);

    LaunchKernel<ARCH35_KEY>(x, y, &tilingData);
}
