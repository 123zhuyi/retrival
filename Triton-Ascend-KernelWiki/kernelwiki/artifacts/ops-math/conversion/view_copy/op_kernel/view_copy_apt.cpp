/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file view_copy.cpp
 * \brief
 */

#include "arch35/view_copy_dim1.h"
#include "arch35/view_copy_dim2.h"
#include "arch35/view_copy_dim3.h"
#include "arch35/view_copy_dim4.h"
#include "arch35/view_copy_dim5.h"
#include "arch35/view_copy_dim8.h"
#include "arch35/view_copy_simt_dim1.h"
#include "arch35/view_copy_simt_dim2.h"
#include "arch35/view_copy_simt_dim3.h"
#include "arch35/view_copy_simt_dim4.h"
#include "arch35/view_copy_simt_dim5.h"
#include "arch35/view_copy_simt_dim8.h"
#include "arch35/view_copy_pure_move_align.h"

using namespace ViewCopy;

#define SIMD_DIM_1_BITWIDTH_1 111
#define SIMD_DIM_1_BITWIDTH_2 112
#define SIMD_DIM_1_BITWIDTH_4 114
#define SIMD_DIM_1_BITWIDTH_8 118
#define SIMD_DIM_2_BITWIDTH_1 121
#define SIMD_DIM_2_BITWIDTH_2 122
#define SIMD_DIM_2_BITWIDTH_4 124
#define SIMD_DIM_2_BITWIDTH_8 128
#define SIMD_DIM_3_BITWIDTH_1 131
#define SIMD_DIM_3_BITWIDTH_2 132
#define SIMD_DIM_3_BITWIDTH_4 134
#define SIMD_DIM_3_BITWIDTH_8 138
#define SIMD_DIM_4_BITWIDTH_1 141
#define SIMD_DIM_4_BITWIDTH_2 142
#define SIMD_DIM_4_BITWIDTH_4 144
#define SIMD_DIM_4_BITWIDTH_8 148
#define SIMD_DIM_5_BITWIDTH_1 151
#define SIMD_DIM_5_BITWIDTH_2 152
#define SIMD_DIM_5_BITWIDTH_4 154
#define SIMD_DIM_5_BITWIDTH_8 158
#define SIMD_DIM_8_BITWIDTH_1 181
#define SIMD_DIM_8_BITWIDTH_2 182
#define SIMD_DIM_8_BITWIDTH_4 184
#define SIMD_DIM_8_BITWIDTH_8 188

#define SIMT_DIM_1_BITWIDTH_1 211
#define SIMT_DIM_1_BITWIDTH_2 212
#define SIMT_DIM_1_BITWIDTH_4 214
#define SIMT_DIM_1_BITWIDTH_8 218
#define SIMT_DIM_2_BITWIDTH_1 221
#define SIMT_DIM_2_BITWIDTH_2 222
#define SIMT_DIM_2_BITWIDTH_4 224
#define SIMT_DIM_2_BITWIDTH_8 228
#define SIMT_DIM_3_BITWIDTH_1 231
#define SIMT_DIM_3_BITWIDTH_2 232
#define SIMT_DIM_3_BITWIDTH_4 234
#define SIMT_DIM_3_BITWIDTH_8 238
#define SIMT_DIM_4_BITWIDTH_1 241
#define SIMT_DIM_4_BITWIDTH_2 242
#define SIMT_DIM_4_BITWIDTH_4 244
#define SIMT_DIM_4_BITWIDTH_8 248
#define SIMT_DIM_5_BITWIDTH_1 251
#define SIMT_DIM_5_BITWIDTH_2 252
#define SIMT_DIM_5_BITWIDTH_4 254
#define SIMT_DIM_5_BITWIDTH_8 258
#define SIMT_DIM_8_BITWIDTH_1 281
#define SIMT_DIM_8_BITWIDTH_2 282
#define SIMT_DIM_8_BITWIDTH_4 284
#define SIMT_DIM_8_BITWIDTH_8 288

#define PURE_MOVE_ALIGNBITWIDTH_1 10001
#define PURE_MOVE_ALIGNBITWIDTH_2 10002
#define PURE_MOVE_ALIGNBITWIDTH_4 10004
#define PURE_MOVE_ALIGNBITWIDTH_8 10008

extern "C" __global__ __aicore__ void view_copy(
    GM_ADDR dst, GM_ADDR dstSize, GM_ADDR dstStride, GM_ADDR dstStorageOffset, GM_ADDR src,GM_ADDR srcSize,
    GM_ADDR srcStride, GM_ADDR srcStorageOffset, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 纯搬运模板 
    // 个位 位宽 1/2/4/8
    if (TILING_KEY_IS(PURE_MOVE_ALIGNBITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ViewCopyTilingDataPureMoveAlign, tilingData, tiling);
        ViewCopy::ViewCopyPureMoveAlign<int8_t> op(tilingData, pipe);
        op.Init(dst, src);
        op.Process();
        return;
    } else if (TILING_KEY_IS(PURE_MOVE_ALIGNBITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ViewCopyTilingDataPureMoveAlign, tilingData, tiling);
        ViewCopy::ViewCopyPureMoveAlign<int16_t> op(tilingData, pipe);
        op.Init(dst, src);
        op.Process();
        return;
    } else if (TILING_KEY_IS(PURE_MOVE_ALIGNBITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ViewCopyTilingDataPureMoveAlign, tilingData, tiling);
        ViewCopy::ViewCopyPureMoveAlign<int32_t> op(tilingData, pipe);
        op.Init(dst, src);
        op.Process();
        return;
    } else if (TILING_KEY_IS(PURE_MOVE_ALIGNBITWIDTH_8)) {
        GET_TILING_DATA_WITH_STRUCT(ViewCopyTilingDataPureMoveAlign, tilingData, tiling);
        ViewCopy::ViewCopyPureMoveAlign<int64_t> op(tilingData, pipe);
        op.Init(dst, src);
        op.Process();
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(ViewCopyTilingData, tilingData, tiling);
    // 百位 SIMD/SIMT 1/2
    // 十位 dim 1/2/3/4/5/8
    // 个位 位宽 1/2/4/8
   if (TILING_KEY_IS(SIMD_DIM_1_BITWIDTH_1)) {
        ViewCopy::ViewCopyDim1<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_1_BITWIDTH_2)) {
        ViewCopy::ViewCopyDim1<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_1_BITWIDTH_4)) {
        ViewCopy::ViewCopyDim1<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_1_BITWIDTH_8)) {
        ViewCopy::ViewCopyDim1<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_2_BITWIDTH_1)) {
        ViewCopy::ViewCopyDim2<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_2_BITWIDTH_2)) {
        ViewCopy::ViewCopyDim2<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_2_BITWIDTH_4)) {
        ViewCopy::ViewCopyDim2<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_2_BITWIDTH_8)) {
        ViewCopy::ViewCopyDim2<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_3_BITWIDTH_1)) {
        ViewCopy::ViewCopyDim3<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_3_BITWIDTH_2)) {
        ViewCopy::ViewCopyDim3<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_3_BITWIDTH_4)) {
        ViewCopy::ViewCopyDim3<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_3_BITWIDTH_8)) {
        ViewCopy::ViewCopyDim3<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_4_BITWIDTH_1)) {
        ViewCopy::ViewCopyDim4<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_4_BITWIDTH_2)) {
        ViewCopy::ViewCopyDim4<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_4_BITWIDTH_4)) {
        ViewCopy::ViewCopyDim4<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_4_BITWIDTH_8)) {
        ViewCopy::ViewCopyDim4<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_5_BITWIDTH_1)) {
        ViewCopy::ViewCopyDim5<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_5_BITWIDTH_2)) {
        ViewCopy::ViewCopyDim5<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_5_BITWIDTH_4)) {
        ViewCopy::ViewCopyDim5<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_5_BITWIDTH_8)) {
        ViewCopy::ViewCopyDim5<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_8_BITWIDTH_1)) {
        ViewCopy::ViewCopyDim8<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_8_BITWIDTH_2)) {
        ViewCopy::ViewCopyDim8<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_8_BITWIDTH_4)) {
        ViewCopy::ViewCopyDim8<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMD_DIM_8_BITWIDTH_8)) {
        ViewCopy::ViewCopyDim8<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_BITWIDTH_1)) {
        ViewCopy::ViewCopySimtDim1<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_BITWIDTH_2)) {
        ViewCopy::ViewCopySimtDim1<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_BITWIDTH_4)) {
        ViewCopy::ViewCopySimtDim1<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_1_BITWIDTH_8)) {
        ViewCopy::ViewCopySimtDim1<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_BITWIDTH_1)) {
        ViewCopy::ViewCopySimtDim2<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_BITWIDTH_2)) {
        ViewCopy::ViewCopySimtDim2<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_BITWIDTH_4)) {
        ViewCopy::ViewCopySimtDim2<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_2_BITWIDTH_8)) {
        ViewCopy::ViewCopySimtDim2<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_BITWIDTH_1)) {
        ViewCopy::ViewCopySimtDim3<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_BITWIDTH_2)) {
        ViewCopy::ViewCopySimtDim3<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_BITWIDTH_4)) {
        ViewCopy::ViewCopySimtDim3<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_3_BITWIDTH_8)) {
        ViewCopy::ViewCopySimtDim3<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_BITWIDTH_1)) {
        ViewCopy::ViewCopySimtDim4<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_BITWIDTH_2)) {
        ViewCopy::ViewCopySimtDim4<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_BITWIDTH_4)) {
        ViewCopy::ViewCopySimtDim4<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_4_BITWIDTH_8)) {
        ViewCopy::ViewCopySimtDim4<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_BITWIDTH_1)) {
        ViewCopy::ViewCopySimtDim5<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_BITWIDTH_2)) {
        ViewCopy::ViewCopySimtDim5<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_BITWIDTH_4)) {
        ViewCopy::ViewCopySimtDim5<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_5_BITWIDTH_8)) {
        ViewCopy::ViewCopySimtDim5<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_BITWIDTH_1)) {
        ViewCopy::ViewCopySimtDim8<int8_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_BITWIDTH_2)) {
        ViewCopy::ViewCopySimtDim8<int16_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_BITWIDTH_4)) {
        ViewCopy::ViewCopySimtDim8<int32_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    } else if (TILING_KEY_IS(SIMT_DIM_8_BITWIDTH_8)) {
        ViewCopy::ViewCopySimtDim8<int64_t> op(pipe, &tilingData);
        op.Init(dst, dstSize, dstStride, dstStorageOffset, src, srcSize, srcStride, srcStorageOffset, out);
        op.Process();
    }
}