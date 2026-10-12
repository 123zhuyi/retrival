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
 * \file as_strided_tiling_key.h
 * \brief as_strided tiling key declare
 */
#ifndef AS_STRIDED_TILING_KEY_H_
#define AS_STRIDED_TILING_KEY_H_

#include <cstdint>

// Path 0: Fully contiguous view, block-copy with DataCopyPad
constexpr uint32_t AS_STRIDED_TILING_KEY_CONTIGUOUS = 0;

// Path 1: Last dimension stride == 1 (contiguous within each row)
constexpr uint32_t AS_STRIDED_TILING_KEY_STRIDE_1 = 1;

// Path 2: General strided last dimension (use DataCopy srcStride)
constexpr uint32_t AS_STRIDED_TILING_KEY_GENERAL_STRIDE = 2;

// Path 3: Last dimension stride == 0 (broadcast / repeat)
constexpr uint32_t AS_STRIDED_TILING_KEY_BROADCAST = 3;

// Path 4: Fallback scalar gather (extreme strides or very small size)
constexpr uint32_t AS_STRIDED_TILING_KEY_SCALAR = 4;

// Path 5: Rank-1 strided copy, split by output elements instead of rows
constexpr uint32_t AS_STRIDED_TILING_KEY_RANK1_STRIDE = 5;

// Path 6: Small non-contiguous stride-1 rows, batch row runs into one copy
constexpr uint32_t AS_STRIDED_TILING_KEY_STRIDE1_ROW_BATCH = 6;

// Path 7: Small general stride, load the covered span then pack useful elements
constexpr uint32_t AS_STRIDED_TILING_KEY_GENERAL_SMALL_SPAN = 7;

// Path 8: Rank-1 stride with bounded span, load contiguous span then pack
constexpr uint32_t AS_STRIDED_TILING_KEY_RANK1_STRIDE_SPAN = 8;

// Path 9: Small aligned contiguous view, single-buffer copy to reduce overhead
constexpr uint32_t AS_STRIDED_TILING_KEY_CONTIGUOUS_SMALL_ALIGNED = 9;

// Path 10: Non-contiguous view whose complete input span fits in UB
constexpr uint32_t AS_STRIDED_TILING_KEY_COMPACT_SPAN = 10;

// Path 11: Compact-span view processed by reusable suffix gather masks
constexpr uint32_t AS_STRIDED_TILING_KEY_COMPACT_SUFFIX = 11;

#endif
