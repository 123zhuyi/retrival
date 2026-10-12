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
 * \file arg_max_with_value_tiling_data.h
 * \brief Tiling data shared by the ArgWithValue host tiling and kernel.
 *
 * The op reduces one axis of an [1,8]-D tensor. The host flattens the logical shape into a canonical
 * 3-D view  firstDim x axisSize x lastDim  (everything before the reduce axis, the axis, everything
 * after) and picks one of three compute patterns:
 *   COPY  (axisSize == 1): the reduce is a no-op; copy input to values, indices are all 0.
 *   LAST  (lastDim  == 1): each output reduces a *contiguous* run of axisSize elements.
 *   NLAST (lastDim  >  1): each output reduces axisSize elements *strided* by lastDim.
 */
#ifndef ARG_MAX_WITH_VALUE_TILING_DATA_H
#define ARG_MAX_WITH_VALUE_TILING_DATA_H

#include <cstdint>

#define ARG_MODE_COPY 0        // axisSize == 1: copy input -> value, index = 0
#define ARG_MODE_LAST 1        // lastDim  == 1: reduce contiguous axis per output row
#define ARG_MODE_NLAST 2       // lastDim  >  1: reduce strided axis per (firstDim, lastDim) output
#define ARG_MODE_LAST_DIRECT 3 // contiguous LAST input handled by the stateless direct kernel

struct ArgMaxWithValueTilingData {
    uint32_t tilingMode; // ARG_SCH_* selected by host; useful for diagnostics, dispatch uses the tiling key
    uint32_t firstDim;   // product of dims before the reduce axis
    uint32_t axisSize;   // length of the reduce axis
    uint32_t lastDim;    // product of dims after the reduce axis
    uint32_t outSize;    // firstDim * lastDim (total output elements)

    uint32_t usedCoreNum; // cores that actually receive work
    uint32_t perCore;     // output elements per core: the BASE block (smaller); see bigCores
    uint32_t bigCores;    // uneven all-core split: the first bigCores cores get perCore+outAlign, rest get perCore
                          // (so all coreNum cores engage even when outSize/coreNum isn't outAlign-aligned).
                          // 0 = uniform split (every core gets perCore) -- COPY / splitAxis / NLAST-batch paths.

    uint32_t rowTile;    // COPY: output elements per tile;  LAST: output rows per tile
    uint32_t innerTile;  // NLAST: lastDim columns processed per output tile
    uint32_t axisTile;   // NLAST: reduce-axis rows loaded per CopyIn
    uint32_t apiTmpSize; // LAST piece path: Broadcast temp bytes (host GetBroadCastMaxMinTmpSize); 0 if unused

    uint32_t splitAxis;   // LAST single-output: 1 = split the reduce axis across cores + cross-core combine
    uint32_t axisPerCore; // LAST split: reduce-axis elements assigned to each core

    uint32_t nlBf;   // NLAST batch: number of consecutive firstDim planes reduced together (1 = per-output path)
    uint32_t nlIPad; // NLAST batch: per-plane UB stride (lastDim rounded up to 32B)
};
#endif // ARG_MAX_WITH_VALUE_TILING_DATA_H
