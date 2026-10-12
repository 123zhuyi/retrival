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
 * \file segsum_tiling_data.h
 * \brief POD tiling data struct shared by arch35 host tiling and kernel (Ascend950).
 *        Per-core batch bounds are derived from blockIdx in the kernel, so the struct
 *        does not carry any core-count sized array.
 */

#ifndef SEGSUM_TILING_DATA_ARCH35_H
#define SEGSUM_TILING_DATA_ARCH35_H

#include <cstdint>

// Elements of x preloaded per chunk on the column-stripe template (TilingKey 0).
constexpr int64_t SEGSUM_X_CHUNK_ARCH35 = 1024;

struct SegsumTilingDataArch35 {
    int64_t needCoreNum = 0;    // cores that actually own batches
    int64_t batches = 0;        // product of all dims but the last one
    int64_t tailDimLength = 0;  // T, length of the last dim of x
    int64_t averageBatches = 0; // batches per core, kernel derives [start, end) from blockIdx
    int64_t rowLen = 0;         // T aligned up to 32B in elements of the input dtype
    int64_t rowNum = 0;         // TilingKey 1: rows computed per block
    int64_t stripeLen = 0;      // TilingKey 0: columns computed per stripe
};

#endif // SEGSUM_TILING_DATA_ARCH35_H
