/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roll_struct.h
 * \brief tiling data struct
 */

#ifndef __ROLL_TILING_DATA_H__
#define __ROLL_TILING_DATA_H__

constexpr uint32_t MAX_DIM_NUM = 8;
constexpr uint32_t BUF_NUM = 2;

// tiling侧计算搬运参数
struct MoveParam {
    int64_t mte3Count = 1;
    int64_t srcOffset[4] = {0};
    int64_t blockCount[4] = {0};
    int64_t blockLen[4] = {0};
    int64_t srcStride[4] = {0};
    int64_t dstOffset[4] = {0};
};

// 切UB参数
struct UbParam {
    int64_t UbSplitAxis = 0;
    int64_t UbCount = 0;
    int64_t UbFactor = 0;
    int64_t UbTailFactor = 0;
};

struct RollTilingData {
    bool isShiftW{false};
    int64_t blockCount;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t blockSplitAxis;
    UbParam mainCoreUbParam;
    UbParam tailCoreUbParam;
    MoveParam moveparam;

    int64_t needCoreNum;
    int64_t perCoreElements;
    int64_t lastCoreElements;
    int64_t basicElements;
    int64_t maxElements;

    int64_t dimNum;
    int64_t shapes[MAX_DIM_NUM];
    int64_t strides[MAX_DIM_NUM];
    int64_t shifts[MAX_DIM_NUM];
};

#endif