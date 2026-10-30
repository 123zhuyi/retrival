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
 * \file sort_with_index_tiling_data.h
 * \brief SortWithIndex tiling data struct (shared by host tiling and device kernel).
 *
 * value/index dtype are NOT carried here: they are dispatched at compile time via the
 * DTYPE_X / DTYPE_INDEX macros (kernel template params). SIZE_MODE is dispatched via the
 * ASCENDC_TPL schMode key. This struct only carries shape / multi-core / padding metadata.
 */

#ifndef SORT_WITH_INDEX_TILING_DATA_H
#define SORT_WITH_INDEX_TILING_DATA_H

struct SortWithIndexTilingData {
    // ---- multi-core row distribution (big/small core) ----
    uint32_t smallCoreRowNum;   // rows handled per small core
    uint32_t bigCoreRowNum;     // rows handled per big core
    uint32_t smallCoreNum;      // number of small cores
    uint32_t bigCoreNum;        // number of big cores
    uint32_t validCoreNum;      // actually launched cores (== SetBlockDim value)
    // ---- shape ----
    uint32_t rowNum;            // total non-sort-axis rows (= prod(shape[:-1]); rank<=1 -> 1)
    uint32_t sliceLen;          // sort-axis length N (= shape[-1])
    uint32_t realSortLen;       // ceil(N, 32) * 32 (Sort granularity)
    uint32_t align8;            // ceil(N, 8) * 8 (Concat granularity)
    uint32_t padLen;            // DataCopyPad in-row tail padding element count (align8 - N)
    uint32_t dupCount;          // Duplicate sentinel fill count (realSortLen - align8)
    // ---- large-axis multi-block merge (SIZE_MODE=1, implemented) ----
    uint32_t tileLen;           // single-block sort length
    uint32_t tileCntPerRow;     // blocks per row (ceil(N / tileLen))
    // ---- attributes ----
    uint32_t axis;              // normalized sort axis (always rank-1 for last-dim; rank<=1 -> 0)
    bool descending;
    bool stable;
};

#endif  // SORT_WITH_INDEX_TILING_DATA_H
