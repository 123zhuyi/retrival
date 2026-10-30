/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file cdist_grad_tiling_data.h
 * \brief CdistGrad TilingData structure definition
 */
#ifndef _CDIST_GRAD_TILING_DATA_H_
#define _CDIST_GRAD_TILING_DATA_H_

struct CdistGradTilingData {
    // Shape parameters
    int64_t batchSize = 0; // B (batch dimension, 1 for 2D input)
    int64_t pSize = 0;     // P (number of rows in x1)
    int64_t rSize = 0;     // R (number of rows in x2)
    int64_t mSize = 0;     // M (feature dimension)

    // Multi-core split parameters
    int64_t usedCoreNum = 0;   // Number of cores used
    int64_t tasksPerCore = 0;  // Tasks per core (each task = one (b,i) pair)
    int64_t tailCoreTasks = 0; // Tasks for the last core

    // UB split parameters
    int64_t mAligned = 0;       // M aligned to 32 bytes (element count)
    int64_t rTile = 0;          // Number of R rows per chunk
    int64_t numRChunks = 0;     // Number of R chunks
    int64_t lastRChunkSize = 0; // Size of last R chunk

    // p value related
    int64_t pModeInt = 0; // p mode (0=p1, 1=p2, 2=pinf, 3=general)
    double pValue = 0.0;  // Actual p value (used in general p path, host-side only)
    float pValueF = 0.0f; // Actual p value as float (used in kernel)

    // Alignment and buffer parameters
    int64_t rTileAligned = 0;     // R_tile aligned to 32 bytes (element count)
    int64_t maskBufSize = 0;      // Compare mask buffer size (bytes)
    int64_t tmpReduceBufSize = 0; // ReduceMax temp buffer size (bytes)
};

#endif // _CDIST_GRAD_TILING_DATA_H_
