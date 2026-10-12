/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RAGGED_BIN_COUNT_TILING_DATA_H
#define RAGGED_BIN_COUNT_TILING_DATA_H

struct RaggedBinCountTilingData {
    int64_t numRows = 0;
    int64_t numSplits = 0;
    int64_t numValues = 0;
    int64_t numBins = 0;
    int64_t outputElements = 0;
    // Number of scheduled logical cores. The kernel uses it for work partitioning and precise-workspace indexes.
    uint32_t usedCoreNum = 0U;
    // Number of float slots each core privatises in UB before touching global memory; 0 disables the
    // privatised path and the kernel scatters straight to GM as before. When non-zero it always equals
    // outputElements, so the private histogram covers the whole output and both mapping modes can use it.
    // The host only sets it when the buffer fits in the dynamic UB budget and the extra write-back traffic
    // is smaller than the global atomics it removes; see PRIVATE_WRITEBACK_FACTOR in the tiling.
    // Occupies the slot that used to be explicit alignment padding, so sizeof() stays 48 bytes and the
    // host/device layout is unchanged.
    uint32_t privateHistElems = 0U;
};

// Bit layout of the schMode tiling key: mappingMode << 2 | binaryOutput << 1 | hasWeights.
// Both sides of the key live on these three constants -- the host tiling encodes with them and the
// kernel entry decodes with them. They used to be host-side constants paired with bare literals in the
// kernel entry, which is the worst kind of duplication here: changing one shift would still compile,
// still emit all eight keys, and only show up as wrong numbers at runtime.
constexpr uint32_t RBC_MAPPING_MODE_SHIFT = 2U;
constexpr uint32_t RBC_BINARY_OUTPUT_SHIFT = 1U;
constexpr uint32_t RBC_HAS_WEIGHTS_MASK = 1U;

#endif // RAGGED_BIN_COUNT_TILING_DATA_H
