/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SPACE_TO_BATCH_ND_TILING_DATA_H_
#define _SPACE_TO_BATCH_ND_TILING_DATA_H_

#include <cstdint>

constexpr int32_t MAX_RANK = 9;
constexpr int32_t MAX_SPATIAL = 7;

struct SpaceToBatchNDTilingData {
    int64_t rank;
    int64_t inShape[MAX_RANK];
    int64_t outShape[MAX_RANK];
    int64_t totalCount;
    int64_t perCoreCount;
    int64_t ubAxis;
    int64_t ubFactor;
    int64_t bufferSize;
    int64_t numSpatialDims;
    int64_t blockShape[MAX_SPATIAL];
    int64_t padTop[MAX_SPATIAL];
    int64_t padBottom[MAX_SPATIAL];
};

struct SpaceToBatchNDCompileInfo {};

#endif
