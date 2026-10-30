/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IS_CLOSE_TILING_DATA_H
#define IS_CLOSE_TILING_DATA_H

#include <stdint.h>

constexpr uint32_t IS_CLOSE_MAX_BROADCAST_DIM = 8;
constexpr uint32_t IS_CLOSE_BROADCAST_MODE_CONTIGUOUS = 0;
constexpr uint32_t IS_CLOSE_BROADCAST_MODE_X1_SCALAR = 1;
constexpr uint32_t IS_CLOSE_BROADCAST_MODE_X2_SCALAR = 2;
constexpr uint32_t IS_CLOSE_BROADCAST_MODE_TAIL_CONTIGUOUS = 3;
constexpr uint32_t IS_CLOSE_BROADCAST_MODE_GENERAL = 4;
constexpr uint32_t IS_CLOSE_TPL_BROADCAST_CONTIGUOUS = 1;
constexpr uint32_t IS_CLOSE_TPL_BROADCAST_X1_SCALAR = 2;
constexpr uint32_t IS_CLOSE_TPL_BROADCAST_X2_SCALAR = 3;
constexpr uint32_t IS_CLOSE_TPL_BROADCAST_TAIL_CONTIGUOUS = 4;
constexpr uint32_t IS_CLOSE_TPL_BROADCAST_GENERAL = 5;
constexpr uint32_t IS_CLOSE_TPL_FP32 = 1;
constexpr uint32_t IS_CLOSE_TPL_FP16 = 2;
constexpr uint32_t IS_CLOSE_TPL_BF16 = 3;
constexpr uint32_t IS_CLOSE_TPL_INT32 = 4;
constexpr uint32_t IS_CLOSE_TPL_DTYPE_COUNT = 4;

struct IsCloseTilingData {
    uint64_t formerCoreNum;
    uint64_t tailCoreNum;
    uint64_t formerCoreDataNum;
    uint64_t tailCoreDataNum;
    uint64_t formerCoreLoopCount;
    uint64_t formerCoreFormerDataNum;
    uint64_t formerCoreTailDataNum;
    uint64_t tailCoreLoopCount;
    uint64_t tailCoreFormerDataNum;
    uint64_t tailCoreTailDataNum;
    uint64_t tileBufferLen;
    float rtol;
    float atol;
    uint32_t equalNan;
    uint32_t rank;
    uint32_t broadcastMode;
    uint32_t reserved;
    uint64_t totalLength;
    uint64_t outShape[IS_CLOSE_MAX_BROADCAST_DIM];
    uint64_t x1Stride[IS_CLOSE_MAX_BROADCAST_DIM];
    uint64_t x2Stride[IS_CLOSE_MAX_BROADCAST_DIM];
};

#endif // IS_CLOSE_TILING_DATA_H
