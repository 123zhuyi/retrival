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
 * \file tile_with_axis_apt.cpp
 * \brief TileWithAxis 算子 kernel 入口（arch35 架构, DAV_3510）
 *
 * TilingKey: UB_AXIS (0/1/2), dtype 用 DTYPE_X.
 * ref: pad_v3_grad_replication_apt.cpp 同模式 (LaunchKernel + 入口两层).
 */

#include "arch35/tile_with_axis_tiling_key.h"
#include "arch35/tile_with_axis.h"

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;

template <int UB_AXIS>
__aicore__ inline void LaunchKernel(GM_ADDR x, GM_ADDR y,
                                     const TileWithAxisTilingData* td)
{
    TileWithAxisKernel<DTYPE_X, UB_AXIS> op;
    op.Init(x, y, td);
    op.Process();
}

template <int UB_AXIS>
__global__ __aicore__ void tile_with_axis(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TileWithAxisTilingData);
    GET_TILING_DATA_WITH_STRUCT(TileWithAxisTilingData, tilingData, tiling);

    LaunchKernel<UB_AXIS>(x, y, &tilingData);
}
