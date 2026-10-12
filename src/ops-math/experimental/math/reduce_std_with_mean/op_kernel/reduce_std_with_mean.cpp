/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_std_with_mean_kernel.h"

template <uint32_t schMode>
__global__ __aicore__ void reduce_std_with_mean(GM_ADDR self, GM_ADDR mean, GM_ADDR output, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReduceStdWithMeanTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceStdWithMeanTilingData, tilingData, tiling);
    NsReduceStdWithMean::ReduceStdWithMean<DTYPE_SELF, schMode> op;
    op.Init(self, mean, output, workspace, &tilingData);
    op.Process();
}
