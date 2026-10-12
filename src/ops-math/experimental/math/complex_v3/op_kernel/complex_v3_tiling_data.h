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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file complex_v3_tiling_data.h
 * \brief ComplexV3 operator TilingData structure definition
 */

#ifndef _COMPLEX_V3_TILING_DATA_H_
#define _COMPLEX_V3_TILING_DATA_H_

// Maximum supported dimensions for broadcast
constexpr uint32_t COMPLEX_V3_MAX_DIM = 8;

struct ComplexV3TilingData {
    // Basic tiling parameters
    int64_t totalLength = 0; // Total number of elements after broadcast
    int64_t blockFactor = 0; // Number of elements per AI Core
    int64_t ubFactor = 0;    // Number of elements per UB iteration

    // Broadcast parameters
    uint32_t broadcastMode = 0; // 0=no broadcast, 1=needs broadcast
    uint32_t dimNum = 0;        // Number of dimensions after broadcast
    uint32_t preloadMode = 1;   // 1=full preload (inputs fit UB), 0=on-demand loading (inputs too large)
    uint32_t reserved0 = 0;     // Padding for alignment
    int64_t realInputSize = 0;  // Actual element count of real input (for UB preload)
    int64_t imagInputSize = 0;  // Actual element count of imag input (for UB preload)

    // Broadcast shape and stride information
    // Used by Kernel to map output linear index back to input indices
    int64_t outShape[COMPLEX_V3_MAX_DIM] = {0};   // Broadcast output shape
    int64_t realStride[COMPLEX_V3_MAX_DIM] = {0}; // real input broadcast stride (0=broadcasted dim)
    int64_t imagStride[COMPLEX_V3_MAX_DIM] = {0}; // imag input broadcast stride (0=broadcasted dim)
};

#endif // _COMPLEX_V3_TILING_DATA_H_
