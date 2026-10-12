/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _BATCH_TO_SPACE_N_D_TILING_DATA_H_
#define _BATCH_TO_SPACE_N_D_TILING_DATA_H_

#include <cstdint>
#include <cstddef>

constexpr size_t MAX_RANK = 8;
constexpr size_t MAX_INPUT_RANK = MAX_RANK + 1;
constexpr size_t MAX_BLOCK_SHAPE_RANK = MAX_INPUT_RANK - 2;
constexpr size_t MAX_EXPAND_RANK = MAX_INPUT_RANK + MAX_BLOCK_SHAPE_RANK;
constexpr size_t CROPS_DIM = 2;

struct B2SNDCompileInfo {};

struct B2SNDInput {
    uint64_t inShape[MAX_INPUT_RANK]{};
    uint64_t blockShape[MAX_BLOCK_SHAPE_RANK]{};
    uint64_t crops[MAX_BLOCK_SHAPE_RANK][CROPS_DIM]{};
    uint64_t outShape[MAX_INPUT_RANK]{};
    uint32_t rank{0};
};

struct B2SNDSimtTilingData {
    B2SNDInput input;
    uint64_t totalBlock;    // 总块数
    uint64_t mainCoreBlock; // 主核块数
    uint32_t needCoreNum;   // 开核数量
    uint32_t mainCoreNum;   // 主核数量
    uint32_t blockSize;     // 每块元素数
    uint32_t tailBlockSize; // 尾块元素数
};

struct B2SNDLargeCTilingData {
    B2SNDInput input;
    uint64_t totalCount;       // 总块数
    uint64_t perCoreCount;     // 每个核上处理的块数
    uint8_t ubAxis;            // 切分轴
    uint32_t ubFactor;         // 切分轴上的大小
    uint32_t outputBufferSize; // 输出 buffer 大小
};

struct B2SNDSmallCTilingData {
    uint64_t oriInShape[MAX_EXPAND_RANK]{};            // 原始x展开的shape
    uint64_t croppedInShape[MAX_EXPAND_RANK]{};        // 预crop后的x展开shape
    uint64_t crops[MAX_BLOCK_SHAPE_RANK][CROPS_DIM]{}; // crops 值
    uint32_t inUbAxis;                                 // 输入切分轴
    uint32_t outUbAxis;                                // 输出切分轴（按输入索引）
    uint32_t inUbFactor;                               // 输入切分轴维度数量
    uint32_t outUbFactor;                              // 输出切分轴维度数量
    uint64_t ubTotalCount;                             // 块数
    uint64_t ubPerCount;                               // 每个核上的块数
    uint32_t coreNum;                                  // 实际核数
    uint32_t ubTileSize;                               // buffer大小
};

#endif // _BATCH_TO_SPACE_N_D_TILING_DATA_H_
