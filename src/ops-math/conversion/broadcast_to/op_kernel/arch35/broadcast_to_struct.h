/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_TO_STRUCT_H_
#define BROADCAST_TO_STRUCT_H_

#include <cstddef>
#include <cstdint>

constexpr size_t BRC_TO_MAX_DMA_DIM_NUM = static_cast<size_t>(0x5);
constexpr size_t BRC_TO_MAX_A_DIM_NUM = static_cast<size_t>(0x8) * 3;
constexpr size_t BRC_TO_MAX_B_DIM_NUM = static_cast<size_t>(0x8) * 2;

struct BroadcastToTilingData {
    int64_t tilingKey;
    int64_t dFactor;
    uint8_t doubleMode;
    uint8_t uAxisCnt;
    uint8_t bufferCnt;
    uint8_t blockAxis;
    uint32_t tensorSize;
    int64_t usedCoreCnt;
    int64_t ntcALen;
    int64_t tcALen;
    int64_t ntcBLen;
    int64_t tcBLen;
    int64_t ntcULen;
    int64_t tcULen;
    int64_t aLpUnit;
    int64_t uLpUnit;
    int64_t uInOffset;
    int64_t uOutOffset;
    int32_t isUNotB;
    int32_t isLastDimB;
    int32_t aAxesNum;
    int32_t bAxesNum;
    uint64_t xSrcStride[BRC_TO_MAX_DMA_DIM_NUM];
    uint32_t xDstStride[BRC_TO_MAX_DMA_DIM_NUM];
    uint32_t xSize[BRC_TO_MAX_DMA_DIM_NUM];
    int64_t aAxesParams[BRC_TO_MAX_A_DIM_NUM];
    int64_t bAxesParams[BRC_TO_MAX_B_DIM_NUM];
};

#endif // BROADCAST_TO_STRUCT_H_
