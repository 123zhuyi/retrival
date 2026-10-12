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
 * \file diag_v2_tiling_data.h
 * \brief DiagV2 TilingData struct definition (arch35)
 *
 * Design basis: DESIGN.md v2.4 Sec 3.2
 * Standard C++ struct (no BEGIN_TILING_DATA_DEF / TILING_KEY_IS macros).
 *
 * Common fields are always valid. 2D→1D fields only valid when IS_1D_INPUT=0.
 * 1D→2D fields only valid when IS_1D_INPUT=1.
 */

#ifndef __DIAG_V2_ARCH35_TILING_DATA_H__
#define __DIAG_V2_ARCH35_TILING_DATA_H__

#include <cstdint>

struct DiagV2Arch35TilingData {
    // === Common fields ===
    int64_t diagonal;       // Diagonal offset k
    int64_t realCoreNum;    // Actual number of cores used
    int64_t tileLength;     // Max elements per tile

    // === 2D→1D fields (IS_1D_INPUT=0) ===
    int64_t xWidth;         // Input matrix width N
    int64_t xHeight;        // Input matrix height M
    int64_t gmOffset;       // GM offset of first diagonal element (linear index)
    int64_t numOut;         // Total number of output elements
    int64_t numPerCore;     // Elements per core (32B-aligned)
    int64_t tailNum;        // Tail element count for the last core
    int64_t threadNum;      // SIMT thread count (≤ 2048)

    // === 1D→2D fields (IS_1D_INPUT=1, mirror DiagFlatArch35TilingData) ===
    int64_t numInput;       // N = numel(x)
    int64_t outWidth;       // W = N + |k|
    int64_t outTotal;       // W * W
    int64_t outPerCore;     // ceil(outTotal / realCoreNum), elements per core
};

#endif // __DIAG_V2_ARCH35_TILING_DATA_H__
