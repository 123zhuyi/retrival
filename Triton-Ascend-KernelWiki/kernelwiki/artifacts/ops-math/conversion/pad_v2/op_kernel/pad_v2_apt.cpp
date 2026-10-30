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
 * \file pad_v2_apt.cpp
 * \brief PadV2 kernel - 复用 PadV3 实现
 */
#include "../pad_v3/arch35/pad_constant.h"
#include "../pad_v3/arch35/pad_slice.h"

using namespace PadV3;

// 复用 PadV3 的 TilingKey 定义
#define CONSTANT_SIMT_BRANCH 20000
#define CONSTANT_SIMT_BIG_SIZE_BRANCH 20001
#define CONSTANT_CUT_LAST_DIM_BRANCH 30010
#define CONSTANT_BIG_LAST_DIM_BRANCH_DIM2 30021
#define CONSTANT_BIG_LAST_DIM_BRANCH_DIM3 30031
#define CONSTANT_BIG_LAST_DIM_BRANCH_DIM4 30041
#define CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM2 30022
#define CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM3 30032
#define CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM4 30042
#define CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM2 30023
#define CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM3 30033
#define CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM4 30043

#define PAD_SLICE_KEY_MOVE_ALIGN 10100
#define PAD_SLICE_KEY_MOVE_ALIGN_LAST_DIM 10101
#define PAD_SLICE_KEY_NDDMA 10102
#define PAD_SLICE_KEY_NDDMA_LAST_DIM 10103
#define PAD_SLICE_KEY_MOVE_ALIGN_TWO_DIM 10150
#define PAD_SLICE_KEY_SIMT 10200
#define PAD_SLICE_KEY_MOVE_ALIGN_GATHER 10300
#define PAD_SLICE_KEY_MOVE_UNALIGN_GATHER 10301
#define PAD_SLICE_KEY_TWO_DIM_SMALL_SHAPE 10400

extern "C" __global__ __aicore__ void pad_v2(GM_ADDR x, GM_ADDR paddings, GM_ADDR constant_values, GM_ADDR y,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_NONE_TILING;

    // 直接调用 PadV3 的 Kernel 函数，传递 constant_values 参数
    if (TILING_KEY_IS(CONSTANT_CUT_LAST_DIM_BRANCH)) { // 30010
        PadV3::LaunchKernelPadWithHugeWidth<DTYPE_X>(x, paddings, y, tiling, constant_values);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM2)) { // 30021
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM2>(x, paddings, y, tiling,
                                                                                          constant_values);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM3)) { // 30031
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM3>(x, paddings, y, tiling,
                                                                                          constant_values);
    } else if (TILING_KEY_IS(CONSTANT_BIG_LAST_DIM_BRANCH_DIM4)) { // 30041
        PadV3::LaunchKernelPadWithNormalWidth<DTYPE_X, CONSTANT_BIG_LAST_DIM_BRANCH_DIM4>(x, paddings, y, tiling,
                                                                                          constant_values);
    } else if (TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM2) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM3) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_GATHER_BRANCH_DIM4)) {
        PadV3::LaunchKernelPadGather<DTYPE_X>(x, paddings, y, tiling, constant_values);
    } else if (TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM2) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM3) ||
               TILING_KEY_IS(CONSTANT_SMALL_LAST_DIM_SCATTER_BRANCH_DIM4)) {
        PadV3::LaunchKernelPadScatter<DTYPE_X>(x, paddings, y, tiling, constant_values);
    } else if (TILING_KEY_IS(CONSTANT_SIMT_BRANCH)) { // 20000
        PadV3::LaunchKernelPadSimt<DTYPE_X>(x, paddings, y, tiling, constant_values);
    } else if (TILING_KEY_IS(CONSTANT_SIMT_BIG_SIZE_BRANCH)) { // 20001
        PadV3::LaunchKernelPadSimtHuge<DTYPE_X>(x, paddings, y, tiling, constant_values);
    } else {
        // Slice 场景处理
        TPipe pipe;
        __gm__ uint8_t* offsets = nullptr;
        __gm__ uint8_t* size = nullptr;
        if (TILING_KEY_IS(PAD_SLICE_KEY_MOVE_ALIGN)) {
            GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignTilingData, tilingData, tiling);
            PadSliceMoveAlignProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_NDDMA)) {
            GET_TILING_DATA_WITH_STRUCT(SliceNDDMATilingData, tilingData, tiling);
            PadSliceNDDMAProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_MOVE_ALIGN_LAST_DIM)) {
            GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignLastDimTilingData, tilingData, tiling);
            PadSliceMoveAlignLastDimProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_NDDMA_LAST_DIM)) {
            GET_TILING_DATA_WITH_STRUCT(SliceNDDMALastDimTilingData, tilingData, tiling);
            PadSliceNDDMALastDimProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_MOVE_ALIGN_TWO_DIM)) {
            GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignLast2DimTilingData, tilingData, tiling);
            PadSliceMoveAlignTwoDimProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_SIMT)) {
            GET_TILING_DATA_WITH_STRUCT(SliceTilingData, tilingData, tiling);
            // 空 tensor 处理
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_MOVE_ALIGN_GATHER)) {
            GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignGatherTilingData, tilingData, tiling);
            PadSliceMoveAlignGatherProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_MOVE_UNALIGN_GATHER)) {
            GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignGatherTilingData, tilingData, tiling);
            PadSliceMoveAlignDataCopyUnalignProcess(x, offsets, size, y, &tilingData, &pipe);
        } else if (TILING_KEY_IS(PAD_SLICE_KEY_TWO_DIM_SMALL_SHAPE)) {
            GET_TILING_DATA_WITH_STRUCT(SliceTwoDimSmallSapeTilingData, tilingData, tiling);
            PadSliceTwoDimSmallShapeProcess(x, offsets, size, y, &tilingData, &pipe);
        }
    }
}
