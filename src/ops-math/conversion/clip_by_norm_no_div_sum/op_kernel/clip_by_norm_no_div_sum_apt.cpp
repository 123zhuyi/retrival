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
 * \file clip_by_norm_no_div_sum_apt.cpp
 * \brief clip_by_norm_no_div_sum Kernel 入口
 */
#include "kernel_operator.h"
#include "arch35/clip_by_norm_no_div_sum_kernel.h"
#include "arch35/clip_by_norm_no_div_sum_tiling_data.h"

using TilingData4 = ClipByNormNoDivSumTilingData<4>; // RANK <= 4
using TilingData8 = ClipByNormNoDivSumTilingData<8>; // RANK > 4

template <int RANK>
__global__ __aicore__ void clip_by_norm_no_div_sum(
    GM_ADDR x, GM_ADDR greater_zeros, GM_ADDR select_ones, GM_ADDR maximum_ones, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GM_ADDR ins[4] = {x, greater_zeros, select_ones, maximum_ones};
    GM_ADDR outs[1] = {y};

    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        ClipByNormNoDivSumKernel<DTYPE_X, 4> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        ClipByNormNoDivSumKernel<DTYPE_X, 8> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}
