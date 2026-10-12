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
 * \file as_strided_tiling_data.h
 * \brief tiling data struct
 */
#ifndef AS_STRIDED_TILING_DATA_H_
#define AS_STRIDED_TILING_DATA_H_

#include <cstdint>

constexpr int64_t AS_STRIDED_MAX_DIMS = 8;

struct AsStridedTilingData {
    // Path selector (see as_strided_tiling_key.h)
    int64_t tilingKey;

    // Output metadata
    int64_t totalOutputElements;
    int64_t outputDimNum;

    // Input metadata
    int64_t inputElementCount;
    int64_t storageOffset;

    // 2D-flattened view: axis_0 = product(all dims except last), axis_1 = last dim
    int64_t lastDimSize;   // size of the innermost dimension
    int64_t lastDimStride; // stride of the innermost dimension
    int64_t axis0Elements; // total number of "rows" (output size / lastDimSize)

    // Multi-core split (by axis_0 rows for paths 1-3, by linear elements for path 0)
    int64_t perCoreElements; // per-core work unit (rows for paths 1-3, elements for path 0)
    int64_t lastCoreElements;
    int64_t usedCoreNum;

    // UB tile configuration
    int64_t ubElements;          // max elements per UB tile transfer
    int64_t blockElements;       // 32B-aligned block element count
    int64_t inputSpanElements;   // complete covered input span for compact-span path
    int64_t suffixStartDim;      // first dim covered by compact suffix mask
    int64_t suffixElements;      // contiguous output elements covered by one suffix mask
    int64_t suffixOuterElements; // number of outer suffix blocks

    // Dimension metadata (for ComputeInputOffset)
    int64_t outSize[AS_STRIDED_MAX_DIMS];
    int64_t outStride[AS_STRIDED_MAX_DIMS];
    int64_t outSizeStride[AS_STRIDED_MAX_DIMS];
};

#endif
