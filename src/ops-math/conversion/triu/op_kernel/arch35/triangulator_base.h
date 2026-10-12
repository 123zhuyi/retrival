/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file triangulator_base.h
 * \brief
 */

#ifndef TRIANGULATOR_BASE_H_
#define TRIANGULATOR_BASE_H_

#include "split_all.h"
#include "split_row_and_col.h"
#include "split_tiny_shape.h"
#include "split_medium_shape.h"

using namespace Triangulator;

#define OUTPUT_ZERO_INT8 1
#define OUTPUT_ZERO_INT16 2
#define OUTPUT_ZERO_INT32 4
#define OUTPUT_ZERO_INT64 8

#define OUTPUT_INPUT_INT8 11
#define OUTPUT_INPUT_INT16 12
#define OUTPUT_INPUT_INT32 14
#define OUTPUT_INPUT_INT64 18

#define OUTPUT_NORMAL_INT8 21
#define OUTPUT_NORMAL_INT16 22
#define OUTPUT_NORMAL_INT32 24
#define OUTPUT_NORMAL_INT64 28

#define TINY_SHAPE_INT8 31
#define TINY_SHAPE_INT16 32
#define TINY_SHAPE_INT32 34
#define TINY_SHAPE_INT64 38

#define SPLIT_MEDIUM_INT8 41
#define SPLIT_MEDIUM_INT16 42
#define SPLIT_MEDIUM_INT32 44
#define SPLIT_MEDIUM_INT64 48

template <typename T, const int32_t MODE>
__aicore__ inline void invokeTemplateSplitAll(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(TriangulatorTilingData, tilingDataPtr, tiling);
    TPipe pipeIn;
    SplitAll<T, MODE> splitAllInstance;
    splitAllInstance.Init(x, y, tilingDataPtr, &pipeIn);
    splitAllInstance.Process();
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void invokeTemplateSplitRowAndCol(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(TriangulatorNormalTilingData, tilingDataPtr, tiling);
    TPipe pipeIn;
    SplitRowAndCol<T, IS_LOWER> splitRowAndColInstance;
    splitRowAndColInstance.Init(x, y, tilingDataPtr, &pipeIn);
    splitRowAndColInstance.Process();
}

template <typename T>
__aicore__ inline void invokeTemplateSplitTinyShape(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(TriangulatorTinyTilingData, tilingDataPtr, tiling);
    TPipe pipeIn;
    SplitTinyShape<T> splitTinyShapeInstance;
    splitTinyShapeInstance.Init(x, y, tilingDataPtr, &pipeIn);
    splitTinyShapeInstance.Process();
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void invokeTemplateSplitMediumShape(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(TriangulatorMediumTilingData, tilingDataPtr, tiling);
    TPipe pipeIn;
    SplitMediumShape<T, IS_LOWER> splitMediumShapeInstance;
    splitMediumShapeInstance.Init(x, y, tilingDataPtr, &pipeIn);
    splitMediumShapeInstance.Process();
}

template <const bool IS_LOWER>
__aicore__ inline void triangulatorEntry(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    // 个位：位宽  1 int8_t    2 int16_t    4 int32_t   8  int64_t
    // 十位：模板  0 output_zero  1 output_input    2 output_normal    3 tiny_shape 4 split_outer
    if (TILING_KEY_IS(OUTPUT_ZERO_INT8)) {
        // output_zero
        invokeTemplateSplitAll<int8_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_ZERO_INT16)) {
        invokeTemplateSplitAll<int16_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_ZERO_INT32)) {
        invokeTemplateSplitAll<int32_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_ZERO_INT64)) {
        invokeTemplateSplitAll<int64_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_INPUT_INT8)) {
        // output_input
        invokeTemplateSplitAll<int8_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_INPUT_INT16)) {
        invokeTemplateSplitAll<int16_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_INPUT_INT32)) {
        invokeTemplateSplitAll<int32_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_INPUT_INT64)) {
        invokeTemplateSplitAll<int64_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_NORMAL_INT8)) {
        // output_normal
        invokeTemplateSplitRowAndCol<int8_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_NORMAL_INT16)) {
        invokeTemplateSplitRowAndCol<int16_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_NORMAL_INT32)) {
        invokeTemplateSplitRowAndCol<int32_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(OUTPUT_NORMAL_INT64)) {
        invokeTemplateSplitRowAndCol<int64_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(TINY_SHAPE_INT8)) {
        // tiny_shape
        invokeTemplateSplitTinyShape<int8_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TINY_SHAPE_INT16)) {
        invokeTemplateSplitTinyShape<int16_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TINY_SHAPE_INT32)) {
        invokeTemplateSplitTinyShape<int32_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TINY_SHAPE_INT64)) {
        invokeTemplateSplitTinyShape<int64_t>(x, y, tiling);
    } else if (TILING_KEY_IS(SPLIT_MEDIUM_INT8)) {
        // split_outer
        invokeTemplateSplitMediumShape<int8_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(SPLIT_MEDIUM_INT16)) {
        invokeTemplateSplitMediumShape<int16_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(SPLIT_MEDIUM_INT32)) {
        invokeTemplateSplitMediumShape<int32_t, IS_LOWER>(x, y, tiling);
    } else if (TILING_KEY_IS(SPLIT_MEDIUM_INT64)) {
        invokeTemplateSplitMediumShape<int64_t, IS_LOWER>(x, y, tiling);
    }
}
#endif // TRIANGULATOR_BASE_H_
