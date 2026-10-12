/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file one_hot_apt.cpp
 * \brief
 */

#define TILING_KEY_WITHOUT_UB 1000
#define TILING_KEY_WITH_UB 1001

#include "kernel_operator.h"
#include "arch35/one_hot.h"
#include "arch35/one_hot_mix.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void one_hot(
    GM_ADDR x, GM_ADDR depth, GM_ADDR on_value, GM_ADDR off_value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    TPipe TPipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(TILING_KEY_WITHOUT_UB)) {
        OneHot::OneHot<DTYPE_X, DTYPE_DEPTH, DTYPE_Y> op;
        op.Init(x, depth, on_value, off_value, y, userWs, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITH_UB)) {
        OneHot::OneHotMix<DTYPE_X, DTYPE_DEPTH, DTYPE_Y> op;
        op.Init(x, depth, on_value, off_value, y, userWs, &TPipe, &tilingData);
        op.Process();
    }
}