/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BATCH_TO_SPACE_TILING_DATA_H_
#define _BATCH_TO_SPACE_TILING_DATA_H_

#include <cstdint>

constexpr size_t AXIS_COUNT = 4;

struct BatchToSpaceTilingData {
    // common fields
    int64_t inShape[AXIS_COUNT];
    int64_t outShape[AXIS_COUNT];
    uint64_t totalCount;
    uint64_t perCoreCount;
    uint32_t ubFactor;
    uint32_t bufferSize;

    // BatchToSpace specific fields
    int64_t blockSize;
    int64_t cropTop;
    int64_t cropBottom;
    int64_t cropLeft;
    int64_t cropRight;
};

struct BatchToSpaceCompileInfo {};

#endif // _BATCH_TO_SPACE_TILING_DATA_H_
