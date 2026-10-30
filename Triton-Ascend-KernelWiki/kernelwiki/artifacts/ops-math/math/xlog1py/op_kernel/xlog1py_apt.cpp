/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// Xlog1py Kernel 入口
#include "kernel_operator.h"
#include "arch35/xlog1py_kernel.h"
#include "arch35/xlog1py_tiling_struct.h"

constexpr int64_t kRank4D = 4;
constexpr int64_t kRank8D = 8;
using TilingData4 = Xlog1pyTilingData<kRank4D>;
using TilingData8 = Xlog1pyTilingData<kRank8D>;

template<int RANK>
__global__ __aicore__ void xlog1py(
    GM_ADDR x, GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[2]  = {x, y};
    GM_ADDR outs[1] = {z};
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        Xlog1pyKernel<DTYPE_X, 4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        Xlog1pyKernel<DTYPE_X, 8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
