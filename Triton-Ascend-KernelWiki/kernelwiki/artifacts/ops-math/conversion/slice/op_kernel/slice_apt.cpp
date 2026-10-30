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
 * \file slice.cpp
 * \brief
 */

#include "arch35/slice_arch35.h"

using namespace Slice;

#define SLICE_KEY_MOVE_ALIGN 100
#define SLICE_KEY_MOVE_ALIGN_LAST_DIM 101
#define SLICE_KEY_NDDMA 102
#define SLICE_KEY_NDDMA_LAST_DIM 103
#define SLICE_KEY_MOVE_ALIGN_TWO_DIM 150
#define SLICE_KEY_SIMT 200
#define SLICE_KEY_MOVE_ALIGN_GATHER 300
#define SLICE_KEY_MOVE_UNALIGN_GATHER 301
#define SLICE_KEY_TWO_DIM_SMALL_SHAPE 400

extern "C" __global__ __aicore__ void slice(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;
    TPipe pipe;
    if (TILING_KEY_IS(SLICE_KEY_MOVE_ALIGN)) {
        GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignTilingData, tilingData, tiling);
        SliceMoveAlignProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_NDDMA)) {
        GET_TILING_DATA_WITH_STRUCT(SliceNDDMATilingData, tilingData, tiling);
        SliceNDDMAProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_MOVE_ALIGN_LAST_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignLastDimTilingData, tilingData, tiling);
        SliceMoveAlignLastDimProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_NDDMA_LAST_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(SliceNDDMALastDimTilingData, tilingData, tiling);
        SliceNDDMALastDimProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_MOVE_ALIGN_TWO_DIM)) {
        GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignLast2DimTilingData, tilingData, tiling);
        SliceMoveAlignTwoDimProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(SliceTilingData, tilingData, tiling);
        // 空tenseor处理
    } else if (TILING_KEY_IS(SLICE_KEY_MOVE_ALIGN_GATHER)) {
        GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignGatherTilingData, tilingData, tiling);
        SliceMoveAlignGatherProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_MOVE_UNALIGN_GATHER)) {
        GET_TILING_DATA_WITH_STRUCT(SliceMoveAlignGatherTilingData, tilingData, tiling);
        SliceMoveAlignDataCopyUnalignProcess(x, offsets, size, y, &tilingData, &pipe);
    } else if (TILING_KEY_IS(SLICE_KEY_TWO_DIM_SMALL_SHAPE)) {
        GET_TILING_DATA_WITH_STRUCT(SliceTwoDimSmallSapeTilingData, tilingData, tiling);
        SliceTwoDimSmallShapeProcess(x, offsets, size, y, &tilingData, &pipe);
    }
}