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
 * \file right_shift_tiling_data.h
 * \brief RightShift tiling data
 */

#ifndef RIGHT_SHIFT_TILING_DATA_H
#define RIGHT_SHIFT_TILING_DATA_H

#include <stdint.h>

constexpr uint32_t RIGHT_SHIFT_MAX_BROADCAST_DIM = 8;
constexpr uint32_t RIGHT_SHIFT_MODE_CONTIGUOUS = 0;
constexpr uint32_t RIGHT_SHIFT_MODE_X_SCALAR = 1;
constexpr uint32_t RIGHT_SHIFT_MODE_Y_SCALAR = 2;
constexpr uint32_t RIGHT_SHIFT_MODE_TAIL_CONTIGUOUS = 3;
constexpr uint32_t RIGHT_SHIFT_MODE_GENERAL = 4;

constexpr uint32_t RIGHT_SHIFT_TPL_INT8 = 1;
constexpr uint32_t RIGHT_SHIFT_TPL_UINT8 = 2;
constexpr uint32_t RIGHT_SHIFT_TPL_INT16 = 3;
constexpr uint32_t RIGHT_SHIFT_TPL_UINT16 = 4;
constexpr uint32_t RIGHT_SHIFT_TPL_INT32 = 5;
constexpr uint32_t RIGHT_SHIFT_TPL_UINT32 = 6;
constexpr uint32_t RIGHT_SHIFT_TPL_INT64 = 7;
constexpr uint32_t RIGHT_SHIFT_TPL_UINT64 = 8;
constexpr uint32_t RIGHT_SHIFT_TPL_DTYPE_COUNT = 8;

struct RightShiftTilingData {
    uint64_t formerCoreNum;
    uint64_t tailCoreNum;
    uint64_t formerCoreDataNum;
    uint64_t tailCoreDataNum;
    uint64_t tileBufferLen;
    uint64_t totalLength;
    uint32_t rank;
    uint32_t mode;
    uint64_t outShape[RIGHT_SHIFT_MAX_BROADCAST_DIM];
    uint64_t xStride[RIGHT_SHIFT_MAX_BROADCAST_DIM];
    uint64_t yStride[RIGHT_SHIFT_MAX_BROADCAST_DIM];
};

#endif // RIGHT_SHIFT_TILING_DATA_H
