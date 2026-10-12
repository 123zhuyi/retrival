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
 * \file cdist_grad_tiling_data_arch22.h
 * \brief Tiling data for CdistGrad AscendC kernel (arch22/910B, all p values)
 *
 * All inputs are the broadcast [B, P, Q, M] continuous tensors prepared by aclnn
 * (grad/cdist/x1/x2 all broadcast to the same shape). Kernel works fully vectorized.
 */

#ifndef CDIST_GRAD_TILING_DATA_H
#define CDIST_GRAD_TILING_DATA_H

#include <cstdint>

struct CdistGradTilingData {
    // Shape (post-broadcast: all inputs share [B, P, Q, M])
    int64_t batchSize = 0; // B
    int64_t pSize = 0;     // P
    int64_t rSize = 0;     // Q
    int64_t mSize = 0;     // M (valid elements per feature row)
    int64_t mAligned = 0;  // full M aligned to 256B in fp32 element count (ws slot stride)

    int64_t mTileSize = 0;
    int64_t numMTiles = 1;
    int64_t lastMTileSize = 0;

    // R-tile (j chunk) parameters
    int64_t rTile = 0;
    int64_t numRChunks = 0;
    int64_t lastRChunkSize = 0;

    int64_t cTile = 1;

    int64_t pTile = 1;

    // Multi-core split along ceil(B*P / pTile) tasks
    int64_t tasksPerCore = 0;
    int64_t tailCoreTasks = 0;
    int64_t usedCoreNum = 0;

    int64_t qSplit = 1;
    int64_t qPartSize = 0; // Q range size per part = CeilDiv(Q, qSplit)

    // Power temporary buffer size in bytes (from GetPowerMaxMinTmpSize on host)
    int64_t tmpBufSize = 0;

    // p value (float, used by p-general Power exponent p-1)
    float pValueF = 2.0f;
};

#endif // CDIST_GRAD_TILING_DATA_H
