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
 * \file strided_slice_grad.cpp
 * \brief
 */

#include "arch35/all_clear.h"
#include "arch35/copy_out_simt.h"
#include "arch35/split_all_axis_move_align.h"
#include "arch35/split_front_axis_move_align.h"
#include "arch35/split_all_axis_nddma.h"
#include "arch35/split_front_axis_nddma.h"

using namespace StridedSliceGrad;

#define ALL_CLEAR_EMPTY_TENSOR_INT8 101
#define ALL_CLEAR_EMPTY_TENSOR_INT16 102
#define ALL_CLEAR_EMPTY_TENSOR_INT32 104
#define ALL_CLEAR_EMPTY_TENSOR_INT64 108

#define ALL_CLEAR_SIMT_INT8 111
#define ALL_CLEAR_SIMT_INT16 112
#define ALL_CLEAR_SIMT_INT32 114
#define ALL_CLEAR_SIMT_INT64 118

#define ALL_CLEAR_SPLIT_FRONT_AXIS_INT8 121
#define ALL_CLEAR_SPLIT_FRONT_AXIS_INT16 122
#define ALL_CLEAR_SPLIT_FRONT_AXIS_INT32 124
#define ALL_CLEAR_SPLIT_FRONT_AXIS_INT64 128

#define ALL_CLEAR_SPLIT_ALL_AXIS_INT8 131
#define ALL_CLEAR_SPLIT_ALL_AXIS_INT16 132
#define ALL_CLEAR_SPLIT_ALL_AXIS_INT32 134
#define ALL_CLEAR_SPLIT_ALL_AXIS_INT64 138

#define NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT8 241
#define NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT16 242
#define NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT32 244
#define NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT64 248

#define NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT8 251
#define NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT16 252
#define NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT32 254
#define NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT64 258

template <typename T>
__aicore__ inline void invokeTemplateAllClearEmptyTensor(GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                                         TPipe& pipeIn)
{
    AllClear<T> allClearInstance;
    allClearInstance.Init(output, tilingData, pipeIn);
    allClearInstance.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateAllClear(GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                              TPipe& pipeIn)
{
    AllClear<T> allClearInstance;
    allClearInstance.Init(output, tilingData, pipeIn);
    allClearInstance.Process();
    allClearInstance.SyncALLCores();
    pipeIn.Reset();
}

template <typename T>
__aicore__ inline void invokeTemplateAllClearSimt(GM_ADDR output, const StridedSliceGradTilingData& tilingData,
                                                  TPipe& pipeIn)
{
    AllClear<T> allClearInstance;
    allClearInstance.Init(output, tilingData, pipeIn);
    allClearInstance.Process();
    allClearInstance.SyncALLCoresSimt();
    pipeIn.Reset();
}

template <typename T>
__aicore__ inline void invokeTemplateCopyOutSimt(GM_ADDR dy, GM_ADDR output,
                                                 const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    CopyOutSimt<T> copyOutSimtInstance;
    copyOutSimtInstance.Init(dy, output, tilingData, pipeIn);
    copyOutSimtInstance.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateSplitFrontAxisMoveAlign(GM_ADDR dy, GM_ADDR output,
                                                             const StridedSliceGradTilingData& tilingData,
                                                             TPipe& pipeIn)
{
    SplitFrontAxisMoveAlign<T> splitFrontAxisMoveAlignInstance;
    splitFrontAxisMoveAlignInstance.Init(dy, output, tilingData, pipeIn);
    splitFrontAxisMoveAlignInstance.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateSplitAllAxisMoveAlign(GM_ADDR dy, GM_ADDR output,
                                                           const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    SplitAllAxisMoveAlign<T> splitAllAxisMoveAlignInstance;
    splitAllAxisMoveAlignInstance.Init(dy, output, tilingData, pipeIn);
    splitAllAxisMoveAlignInstance.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateSplitFrontAxisNddma(GM_ADDR dy, GM_ADDR output,
                                                         const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    SplitFrontAxisNddma<T> splitFrontAxisNddmaInstance;
    splitFrontAxisNddmaInstance.Init(dy, output, tilingData, pipeIn);
    splitFrontAxisNddmaInstance.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateSplitAllAxisNddma(GM_ADDR dy, GM_ADDR output,
                                                       const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    SplitAllAxisNddma<T> splitAllAxisNddmaInstance;
    splitAllAxisNddmaInstance.Init(dy, output, tilingData, pipeIn);
    splitAllAxisNddmaInstance.Process();
}

extern "C" __global__ __aicore__ void strided_slice_grad(GM_ADDR shape, GM_ADDR begin, GM_ADDR end, GM_ADDR strides,
                                                         GM_ADDR dy, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe pipeIn;
    // 个位：位宽  1 int8_t  2 int16_t         4 int32_t        8  int64_t
    // 十位：模板  0 EmptyTensor  1 SIMT    2 SplitFrontAxis  3 SplitAllAxis   4 SplitFrontAxisNddma  5
    // SplitAllAxisNddma 百位：清零  1 ALL_CLEAR    2 NO_CLEAR 千位：DIM特化模板     1  dim_1    2  dim_2   0 dim_other
    if (TILING_KEY_IS(ALL_CLEAR_EMPTY_TENSOR_INT8)) {
        // EmptyTensor
        invokeTemplateAllClearEmptyTensor<int8_t>(output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_EMPTY_TENSOR_INT16)) {
        invokeTemplateAllClearEmptyTensor<int16_t>(output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_EMPTY_TENSOR_INT32)) {
        invokeTemplateAllClearEmptyTensor<int32_t>(output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_EMPTY_TENSOR_INT64)) {
        invokeTemplateAllClearEmptyTensor<int64_t>(output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SIMT_INT8)) {
        // Simt
        invokeTemplateAllClearSimt<int8_t>(output, tilingData, pipeIn);
        invokeTemplateCopyOutSimt<int8_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SIMT_INT16)) {
        invokeTemplateAllClearSimt<int16_t>(output, tilingData, pipeIn);
        invokeTemplateCopyOutSimt<int16_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SIMT_INT32)) {
        invokeTemplateAllClearSimt<int32_t>(output, tilingData, pipeIn);
        invokeTemplateCopyOutSimt<int32_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SIMT_INT64)) {
        invokeTemplateAllClearSimt<int64_t>(output, tilingData, pipeIn);
        invokeTemplateCopyOutSimt<int64_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_FRONT_AXIS_INT8)) {
        // SplitFrontAxis
        invokeTemplateAllClear<int8_t>(output, tilingData, pipeIn);
        invokeTemplateSplitFrontAxisMoveAlign<int8_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_FRONT_AXIS_INT16)) {
        invokeTemplateAllClear<int16_t>(output, tilingData, pipeIn);
        invokeTemplateSplitFrontAxisMoveAlign<int16_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_FRONT_AXIS_INT32)) {
        invokeTemplateAllClear<int32_t>(output, tilingData, pipeIn);
        invokeTemplateSplitFrontAxisMoveAlign<int32_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_FRONT_AXIS_INT64)) {
        invokeTemplateAllClear<int64_t>(output, tilingData, pipeIn);
        invokeTemplateSplitFrontAxisMoveAlign<int64_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_ALL_AXIS_INT8)) {
        // SplitAllAxis
        invokeTemplateAllClear<int8_t>(output, tilingData, pipeIn);
        invokeTemplateSplitAllAxisMoveAlign<int8_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_ALL_AXIS_INT16)) {
        invokeTemplateAllClear<int16_t>(output, tilingData, pipeIn);
        invokeTemplateSplitAllAxisMoveAlign<int16_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_ALL_AXIS_INT32)) {
        invokeTemplateAllClear<int32_t>(output, tilingData, pipeIn);
        invokeTemplateSplitAllAxisMoveAlign<int32_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(ALL_CLEAR_SPLIT_ALL_AXIS_INT64)) {
        invokeTemplateAllClear<int64_t>(output, tilingData, pipeIn);
        invokeTemplateSplitAllAxisMoveAlign<int64_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT8)) {
        // SplitFrontAxisNddma
        invokeTemplateSplitFrontAxisNddma<int8_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT16)) {
        invokeTemplateSplitFrontAxisNddma<int16_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT32)) {
        invokeTemplateSplitFrontAxisNddma<int32_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_FRONT_AXIS_NDDMA_INT64)) {
        invokeTemplateSplitFrontAxisNddma<int64_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT8)) {
        // SplitAllAxisNddma
        invokeTemplateSplitAllAxisNddma<int8_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT16)) {
        invokeTemplateSplitAllAxisNddma<int16_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT32)) {
        invokeTemplateSplitAllAxisNddma<int32_t>(dy, output, tilingData, pipeIn);
    } else if (TILING_KEY_IS(NO_CLEAR_SPLIT_ALL_AXIS_NDDMA_INT64)) {
        invokeTemplateSplitAllAxisNddma<int64_t>(dy, output, tilingData, pipeIn);
    }
}