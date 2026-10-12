/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef _SLICE_WITH_AXES_TILING_DATA_H_
#define _SLICE_WITH_AXES_TILING_DATA_H_

#include <cstdint>

constexpr size_t MAX_AXIS_COUNT = 8;

struct SliceWithAxesTilingData {
    int64_t inShape[MAX_AXIS_COUNT];
    int64_t outShape[MAX_AXIS_COUNT];
    int64_t fullOffsets[MAX_AXIS_COUNT];
    uint64_t totalCount;
    uint64_t perCoreCount;
    uint8_t ubAxis;
    uint32_t ubFactor;
    uint32_t bufferSize;
    uint8_t rank;
};

struct SliceWithAxesCompileInfo {};

#endif // _SLICE_WITH_AXES_TILING_DATA_H_
