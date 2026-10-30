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
 * \file trilu_entry.h
 * \brief triluEntry: dispatch to triangulator invoke functions based on tiling key.
 *
 * trilu reuses triangulator's SIMD kernel (SplitAll / SplitRowAndCol /
 * SplitTinyShape / SplitMediumShape).  The tiling key encodes:
 *   hundreds digit = upper (0=tril, 1=triu)
 *   tens+units      = baseKey (tilingMode*10 + dtypeBytes)
 *
 * Each full tiling key is defined as a macro (TRIL_/TRIU_) and dispatched
 * via TILING_KEY_IS.  For NORMAL and MEDIUM modes the IS_LOWER template
 * parameter is selected from the key (tril -> true, triu -> false).
 */

#ifndef TRILU_ENTRY_H_
#define TRILU_ENTRY_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../triu/arch35/triangulator_base.h"

namespace NsTrilu {
using namespace AscendC;
using namespace Triangulator;

// tril: hundreds digit = 0  (key = baseKey)
#define TRIL_OUTPUT_ZERO_INT8 1
#define TRIL_OUTPUT_ZERO_INT16 2
#define TRIL_OUTPUT_ZERO_INT32 4
#define TRIL_OUTPUT_ZERO_INT64 8

#define TRIL_OUTPUT_INPUT_INT8 11
#define TRIL_OUTPUT_INPUT_INT16 12
#define TRIL_OUTPUT_INPUT_INT32 14
#define TRIL_OUTPUT_INPUT_INT64 18

#define TRIL_OUTPUT_NORMAL_INT8 21
#define TRIL_OUTPUT_NORMAL_INT16 22
#define TRIL_OUTPUT_NORMAL_INT32 24
#define TRIL_OUTPUT_NORMAL_INT64 28

#define TRIL_TINY_SHAPE_INT8 31
#define TRIL_TINY_SHAPE_INT16 32
#define TRIL_TINY_SHAPE_INT32 34
#define TRIL_TINY_SHAPE_INT64 38

#define TRIL_SPLIT_MEDIUM_INT8 41
#define TRIL_SPLIT_MEDIUM_INT16 42
#define TRIL_SPLIT_MEDIUM_INT32 44
#define TRIL_SPLIT_MEDIUM_INT64 48

// triu: hundreds digit = 1  (key = 100 + baseKey)
#define TRIU_OUTPUT_ZERO_INT8 101
#define TRIU_OUTPUT_ZERO_INT16 102
#define TRIU_OUTPUT_ZERO_INT32 104
#define TRIU_OUTPUT_ZERO_INT64 108

#define TRIU_OUTPUT_INPUT_INT8 111
#define TRIU_OUTPUT_INPUT_INT16 112
#define TRIU_OUTPUT_INPUT_INT32 114
#define TRIU_OUTPUT_INPUT_INT64 118

#define TRIU_OUTPUT_NORMAL_INT8 121
#define TRIU_OUTPUT_NORMAL_INT16 122
#define TRIU_OUTPUT_NORMAL_INT32 124
#define TRIU_OUTPUT_NORMAL_INT64 128

#define TRIU_TINY_SHAPE_INT8 131
#define TRIU_TINY_SHAPE_INT16 132
#define TRIU_TINY_SHAPE_INT32 134
#define TRIU_TINY_SHAPE_INT64 138

#define TRIU_SPLIT_MEDIUM_INT8 141
#define TRIU_SPLIT_MEDIUM_INT16 142
#define TRIU_SPLIT_MEDIUM_INT32 144
#define TRIU_SPLIT_MEDIUM_INT64 148

__aicore__ inline void triluEntry(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    // 个位：位宽  1 int8_t    2 int16_t    4 int32_t   8 int64_t
    // 十位：模板  0 output_zero  1 output_input    2 output_normal    3 tiny_shape  4 split_medium
    // 百位：方向  0 tril(isLower=true)   1 triu(isLower=false)
    if (TILING_KEY_IS(TRIL_OUTPUT_ZERO_INT8) || TILING_KEY_IS(TRIU_OUTPUT_ZERO_INT8)) {
        // output_zero
        invokeTemplateSplitAll<int8_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_ZERO_INT16) || TILING_KEY_IS(TRIU_OUTPUT_ZERO_INT16)) {
        invokeTemplateSplitAll<int16_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_ZERO_INT32) || TILING_KEY_IS(TRIU_OUTPUT_ZERO_INT32)) {
        invokeTemplateSplitAll<int32_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_ZERO_INT64) || TILING_KEY_IS(TRIU_OUTPUT_ZERO_INT64)) {
        invokeTemplateSplitAll<int64_t, OUTPUT_ZERO_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_INPUT_INT8) || TILING_KEY_IS(TRIU_OUTPUT_INPUT_INT8)) {
        // output_input
        invokeTemplateSplitAll<int8_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_INPUT_INT16) || TILING_KEY_IS(TRIU_OUTPUT_INPUT_INT16)) {
        invokeTemplateSplitAll<int16_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_INPUT_INT32) || TILING_KEY_IS(TRIU_OUTPUT_INPUT_INT32)) {
        invokeTemplateSplitAll<int32_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_INPUT_INT64) || TILING_KEY_IS(TRIU_OUTPUT_INPUT_INT64)) {
        invokeTemplateSplitAll<int64_t, OUTPUT_INPUT_MODE>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_NORMAL_INT8)) {
        // output_normal
        invokeTemplateSplitRowAndCol<int8_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_OUTPUT_NORMAL_INT8)) {
        invokeTemplateSplitRowAndCol<int8_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_NORMAL_INT16)) {
        invokeTemplateSplitRowAndCol<int16_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_OUTPUT_NORMAL_INT16)) {
        invokeTemplateSplitRowAndCol<int16_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_NORMAL_INT32)) {
        invokeTemplateSplitRowAndCol<int32_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_OUTPUT_NORMAL_INT32)) {
        invokeTemplateSplitRowAndCol<int32_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_OUTPUT_NORMAL_INT64)) {
        invokeTemplateSplitRowAndCol<int64_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_OUTPUT_NORMAL_INT64)) {
        invokeTemplateSplitRowAndCol<int64_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_TINY_SHAPE_INT8) || TILING_KEY_IS(TRIU_TINY_SHAPE_INT8)) {
        // tiny_shape
        invokeTemplateSplitTinyShape<int8_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_TINY_SHAPE_INT16) || TILING_KEY_IS(TRIU_TINY_SHAPE_INT16)) {
        invokeTemplateSplitTinyShape<int16_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_TINY_SHAPE_INT32) || TILING_KEY_IS(TRIU_TINY_SHAPE_INT32)) {
        invokeTemplateSplitTinyShape<int32_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_TINY_SHAPE_INT64) || TILING_KEY_IS(TRIU_TINY_SHAPE_INT64)) {
        invokeTemplateSplitTinyShape<int64_t>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_SPLIT_MEDIUM_INT8)) {
        // split_medium
        invokeTemplateSplitMediumShape<int8_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_SPLIT_MEDIUM_INT8)) {
        invokeTemplateSplitMediumShape<int8_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_SPLIT_MEDIUM_INT16)) {
        invokeTemplateSplitMediumShape<int16_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_SPLIT_MEDIUM_INT16)) {
        invokeTemplateSplitMediumShape<int16_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_SPLIT_MEDIUM_INT32)) {
        invokeTemplateSplitMediumShape<int32_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_SPLIT_MEDIUM_INT32)) {
        invokeTemplateSplitMediumShape<int32_t, false>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIL_SPLIT_MEDIUM_INT64)) {
        invokeTemplateSplitMediumShape<int64_t, true>(x, y, tiling);
    } else if (TILING_KEY_IS(TRIU_SPLIT_MEDIUM_INT64)) {
        invokeTemplateSplitMediumShape<int64_t, false>(x, y, tiling);
    }
}

} // namespace NsTrilu
#endif // TRILU_ENTRY_H_
