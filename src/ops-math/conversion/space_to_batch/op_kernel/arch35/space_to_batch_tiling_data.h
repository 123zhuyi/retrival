/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SPACE_TO_BATCH_TILING_DATA_H_
#define _SPACE_TO_BATCH_TILING_DATA_H_

#include <cstdint>
#include <cstddef>

constexpr size_t STB_AXIS_COUNT = 4;
constexpr size_t STB_AXIS_N = 0;
constexpr size_t STB_AXIS_H = 1;
constexpr size_t STB_AXIS_W = 2;
constexpr size_t STB_AXIS_C = 3;

struct SpaceToBatchCompileInfo {};

struct SpaceToBatchTilingData {
    int64_t inShape[STB_AXIS_COUNT];  // 输入 shape [N, H_in, W_in, C]
    int64_t outShape[STB_AXIS_COUNT]; // 输出 shape [N*bs*bs, H_out, W_out, C]
    int64_t blockSize;                // bs
    int64_t paddings[2][2];           // [[pad_top, pad_bottom], [pad_left, pad_right]]
    uint64_t totalCount;              // UB 切分矩阵块总数
    uint64_t perCoreCount;            // 每核处理的矩阵块数
    uint8_t ubAxis;                   // UB 切分轴 (0=N, 1=H, 2=W, 3=C)
    uint32_t ubFactor;                // 切分因子
    uint32_t bufferSize;              // UB buffer 字节大小
};

#endif // _SPACE_TO_BATCH_TILING_DATA_H_
