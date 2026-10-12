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
 * \file cross_struct.h
 * \brief cross tiling data
 */

#ifndef CROSS_STRUCT_H
#define CROSS_STRUCT_H

#include <cstdint>

constexpr int64_t MAX_DIM = 8;

namespace CrossConst {
// Tiling policy (see cross_tiling.cpp DoOpTiling).
//
// MIN_VECTORS_PER_BLOCK: empirically tuned workload floor below which per-block
// launch overhead (20-30%) dominates the multi-block-per-core benefit. Origin:
// commit 12da5f2fb "perf(cross): adaptive multi-block-per-core tiling for large
// DRAM-bound shapes".
constexpr int64_t MIN_VECTORS_PER_BLOCK = 250000;

// MAX_BLOCKS_PER_CORE: top tier of the block-per-core ladder. The medium tier
// (MEDIUM_BLOCKS_PER_CORE) is half of this.
constexpr int64_t MAX_BLOCKS_PER_CORE = 4;
constexpr int64_t MEDIUM_BLOCKS_PER_CORE = 2;

// MAX_ACTIVE_DIMS: number of non-`dim` dims that the SIMT kernel iterates over.
// Excludes `dim` itself (the size-3 cross-product dim), so MAX_DIM - 1.
constexpr int64_t MAX_ACTIVE_DIMS = MAX_DIM - 1;
} // namespace CrossConst

#pragma pack(push, 8)
struct CrossRegbaseTilingData {
    int64_t totalVectors;
    int64_t coreNum;
    int64_t dim;
    int64_t dimNum;
    int64_t mergedStride[MAX_DIM];
    int64_t x1Stride[MAX_DIM];
    int64_t x2Stride[MAX_DIM];
    int64_t yStride[MAX_DIM];
    int64_t dimStride;
    // Multi-block-per-core tiling: total blocks = coreNum * blocksPerCore,
    // capped at totalVectors. Each block does vectorsPerBlock (+1 for the
    // first 'formerBlock' blocks). This lets the SIMT scheduler balance
    // work across cores via a finer-grained work queue.
    int64_t blocksPerCore;
    int64_t formerBlock;
    int64_t vectorsPerBlock;
    int64_t usedInt64;
    // Broadcast-aware normalization: indices of dims with mergedShape > 1 and i != dim.
    // The kernel iterates only over these dims, skipping mergedShape=1 (broadcast) dims.
    int64_t activeDimCount;
    int64_t activeDimIndices[MAX_DIM];
};
#pragma pack(pop)

#endif
