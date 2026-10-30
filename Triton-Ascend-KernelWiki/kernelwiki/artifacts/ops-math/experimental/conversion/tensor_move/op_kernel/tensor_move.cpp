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
 * \file tensor_move.cpp
 * \brief TensorMove kernel entry.
 */

#include "tensor_move.h"
#include "tensor_move_tiling_key.h"

using namespace NsTensorMove;

template <uint32_t schMode>
__global__ __aicore__ void tensor_move(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(TensorMoveTilingData);
    GET_TILING_DATA_WITH_STRUCT(TensorMoveTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (schMode == TENSOR_MOVE_TPL_SCH_MODE_0) {
        TensorMoveKernel<int8_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if constexpr (schMode == TENSOR_MOVE_TPL_SCH_MODE_1) {
        TensorMoveKernel<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if constexpr (schMode == TENSOR_MOVE_TPL_SCH_MODE_2) {
        TensorMoveKernel<int32_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if constexpr (schMode == TENSOR_MOVE_TPL_SCH_MODE_3) {
        TensorMoveKernel<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}
