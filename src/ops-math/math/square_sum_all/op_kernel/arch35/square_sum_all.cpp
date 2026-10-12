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
 * \file square_sum_all.cpp
 * \brief SquareSumAll arch35 kernel entry.
 */

#include "square_sum_all.h"
#include "square_sum_all_gpu_aligned.h"
#include "square_sum_all_tiling_key.h"

using namespace SquareSumAllOps;

template <uint64_t KERNEL_MODE>
__global__ __aicore__ void square_sum_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                          GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC || workspace == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(SquareSumAllTilingData);
    GET_TILING_DATA_WITH_STRUCT(SquareSumAllTilingData, tilingData, tiling);
    TPipe pipe;
    if constexpr (KERNEL_MODE == 0) {
        SquareSumAllKernel op;
        op.Init(x1, x2, y1, y2, userWorkspace, &tilingData, &pipe);
        op.Process();
    } else {
        SquareSumAllGpuAligned::SquareSumAllGpuAlignedKernel op;
        op.Init(x1, x2, y1, y2, userWorkspace, &tilingData, &pipe);
        op.Process();
    }
    pipe.Destroy();
}
