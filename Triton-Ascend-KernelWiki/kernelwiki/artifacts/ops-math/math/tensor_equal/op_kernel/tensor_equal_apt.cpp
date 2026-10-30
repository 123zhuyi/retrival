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
 * \file tensor_equal_apt.cpp
 * \brief
 */

#include "arch35/tensor_equal.h"

#define TILING_KEY_NORMAL 121
#define TILING_KEY_DIFF_SHAPE 111
#define TILING_KEY_EMPTY_SHAPE 101

using namespace TensorEqual;

extern "C" __global__ __aicore__ void tensor_equal(GM_ADDR input_x, GM_ADDR input_y, GM_ADDR output_z, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(TILING_KEY_NORMAL)) {
        TensorEqual::TensorEqualKernel<DTYPE_INPUT_X> op(tilingData, pipe);
        op.Init(input_x, input_y, output_z, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DIFF_SHAPE)) {
        TensorEqual::TensorEqualKernel<DTYPE_INPUT_X> op(tilingData, pipe);
        op.Init(input_x, input_y, output_z, userWS);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_EMPTY_SHAPE)) {
        TensorEqual::TensorEqualKernel<DTYPE_INPUT_X> op(tilingData, pipe);
        op.Init(input_x, input_y, output_z, userWS);
        op.Process();
    }
}