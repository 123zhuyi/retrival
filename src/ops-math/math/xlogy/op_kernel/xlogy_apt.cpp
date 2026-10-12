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
 * \file xlogy_apt.cpp
 * \brief Xlogy Kernel 入口
 */
#include "kernel_operator.h"
#include "arch35/xlogy_kernel.h"
#include "arch35/xlogy_tiling_struct.h"

constexpr int64_t kRank4D = 4;
constexpr int64_t kRank8D = 8;
using TilingData4 = XlogyTilingData<kRank4D>;
using TilingData8 = XlogyTilingData<kRank8D>;

template<int RANK>
__global__ __aicore__ void xlogy(
    GM_ADDR x1, GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR ins[2]  = {x1, x2};
    GM_ADDR outs[1] = {y};
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        XlogyKernel<DTYPE_X1, 4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        XlogyKernel<DTYPE_X1, 8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
