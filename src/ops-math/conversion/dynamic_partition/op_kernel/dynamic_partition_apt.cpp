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
 * \file dynamic_partition.cpp
 * \brief DynamicPartition implementation
 */

#include <type_traits>

#include "arch35/dynamic_partition_with_h_mc.h"
#include "arch35/dynamic_partition_with_h_mc_small_w.h"
#include "arch35/dynamic_partition_with_w_mc.h"
#include "arch35/dynamic_partition_with_x_empty.h"
#include "arch35/dynamic_partition_with_xp_empty.h"
#include "arch35/dynamic_partition_with_xp_scalar.h"
#include "arch35/dynamic_partition_with_num_part_one.h"

using namespace AscendC;
using namespace DynPart;

extern "C" __global__ __aicore__ void dynamic_partition(GM_ADDR x, GM_ADDR partitions, GM_ADDR y, GM_ADDR yshape,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr auto b8 = sizeof(uint8_t);
    constexpr auto b16 = sizeof(uint16_t);
    constexpr auto b32 = sizeof(uint32_t);
    constexpr auto b64 = sizeof(uint64_t);
    constexpr auto tSize = sizeof(DTYPE_X);
    using DTYPE_X_ = std::conditional_t<
        tSize != b32,
        std::conditional_t<
            tSize == b8, uint8_t,
            std::conditional_t<tSize == b16, uint16_t, std::conditional_t<tSize == b64, uint64_t, DTYPE_X>>>,
        DTYPE_X>;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;
    SetSysWorkspace(workspace);

    TPipe pipe;
    if (TILING_KEY_IS(KEY_H_MC_UB_CAN_HOLD_SPLIT_W)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        KERNEL_TASK_TYPE(KEY_H_MC_UB_CAN_HOLD_SPLIT_W, KERNEL_TYPE_MIX_AIV_1_0);
        DynPartWithHMC<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_H_MC_WITH_SMALL_W)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        KERNEL_TASK_TYPE(KEY_H_MC_WITH_SMALL_W, KERNEL_TYPE_MIX_AIV_1_0);
        DynPartWithHMCSMALLW<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_H_MC_UB_CANNOT_HOLD_SPLIT_W)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        KERNEL_TASK_TYPE(KEY_H_MC_UB_CANNOT_HOLD_SPLIT_W, KERNEL_TYPE_MIX_AIV_1_0);
        DynPartWithHMC<DTYPE_X_, true> op;
        op.Init(x, partitions, y, yshape, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_W_MC_UB_CAN_HOLD_SPLIT_W)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        DynPartWithWMC<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_W_MC_UB_CANNOT_HOLD_SPLIT_W)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        DynPartWithWMC<DTYPE_X_, true> op;
        op.Init(x, partitions, y, yshape, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_XP_SCALAR)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        DynPartWithXPScalar<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_XP_EMPTY)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        DynPartWithXPEmpty<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_X_EMPTY)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartTilingData, tilingData, tiling);
        KERNEL_TASK_TYPE(KEY_X_EMPTY, KERNEL_TYPE_MIX_AIV_1_0);
        DynPartWithXEmpty<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(KEY_NUM_PART_ONE)) {
        GET_TILING_DATA_WITH_STRUCT(DynPartNumPartOneTilingData, tilingDataNumPartOne, tiling);
        DynPartWithNumPartOne<DTYPE_X_> op;
        op.Init(x, partitions, y, yshape, &tilingDataNumPartOne, &pipe);
        op.Process();
    }
}
