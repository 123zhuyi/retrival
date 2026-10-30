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
 * \file tensor_move.cpp
 * \brief
 */

#include "arch35/tensor_move.h"

#define TILING_KEY_ONE_BYTE 1
#define TILING_KEY_TWO_BYTE 2
#define TILING_KEY_FOUR_BYTE 4
#define TILING_KEY_EIGHT_BYTE 8

using namespace TensorMove;

extern "C" __global__ __aicore__ void tensor_move(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    // tilingkey对应位宽 1/2/4/8
    if (TILING_KEY_IS(TILING_KEY_ONE_BYTE)) {
        TensorMove::TensorMoveKernel<int8_t> op;
        op.Init(x, y, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_TWO_BYTE)) {
        TensorMove::TensorMoveKernel<int16_t> op;
        op.Init(x, y, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FOUR_BYTE)) {
        TensorMove::TensorMoveKernel<int32_t> op;
        op.Init(x, y, userWS, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_EIGHT_BYTE)) {
        TensorMove::TensorMoveKernel<int64_t> op;
        op.Init(x, y, userWS, tilingData);
        op.Process();
    }
}