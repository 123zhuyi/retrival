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

#ifndef _SLICE_LAST_DIM_TILING_DATA_H_
#define _SLICE_LAST_DIM_TILING_DATA_H_

#include <cstdint>

struct SliceLastDimTilingData {
    int64_t outerSize;
    int64_t lastDimIn;
    int64_t lastDimOut;
    int64_t start;
    int64_t stride;
    uint64_t totalCount;
    uint64_t perCoreCount;
    uint32_t ubFactor;
    uint32_t bufferSize;
};

struct SliceLastDimCompileInfo {};

#endif
