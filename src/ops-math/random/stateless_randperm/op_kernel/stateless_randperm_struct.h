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
 * \file stateless_randperm_struct.h
 * \brief tiling base data
 */

#ifndef STATELESS_RANDPERM_STRUCT_H
#define STATELESS_RANDPERM_STRUCT_H

constexpr uint32_t SUB_N_TILE_COUNT = 8;
constexpr uint16_t PHILOX_KEY_SIZE = 2;

struct SortRegBaseTilingDataForRandperm {
    uint32_t numTileDataSize; // h轴ub一次处理个数
    uint32_t unsortedDimParallel; // b轴使用的核数
    uint32_t lastDimTileNum; // h轴循环次数
    uint32_t sortLoopTimes; // b轴循环次数
    uint32_t lastDimNeedCore; // h轴需要的核数
    // keyParamsxxx 预留参数
    // keyParams0 radix分支表示globalHistGmWk_使用核数，radix_one_core代表inqueX的ub大小,merge sort 表示一次处理几个h
    uint32_t keyParams0;
    uint32_t keyParams1; // radix分支清零excusiveBinsGmWk_的核， radix_one_core y2OutQue需要的ub大小, merge xQue ub大小
    uint32_t keyParams2; // radix分支清零的一次ub数据量, radix_one_core 输出int64时，一半的ub偏移, merge y2OutQue的ub大小
    uint32_t keyParams3; // radix分支清零globalHistGmWk_ ub循环次数, merge表示32个数对齐的alginH
    uint32_t keyParams4; // 清零excusiveBinsGmWk_, 每个核处理多少个数
    uint32_t keyParams5; // 清零globalHistGmWk_，每个核处理多少
    uint32_t tmpUbSize; // sort高级api需要的临时ub大小
    int64_t lastAxisNum; // h轴大小
    int64_t unsortedDimNum; // b轴大小
};

struct StatelessRandpermTilingData{
    int32_t randomBits;                     // 用来决定生成随机数的数据类型
    uint32_t islandFactor;                  // 寻岛洗牌：每个AiCore的线程循环次数
    uint32_t islandFactorTail;              // 寻岛洗牌：最后一个AiCore的线程循环次数
    uint32_t castFactor;                    // 类型转换：每个AiCore的线程循环次数
    uint32_t castFactorTail;                // 类型转换：最后一个AiCore的线程循环次数
    uint32_t realCoreNum;                   // 实际需要的核数，由sort决定，因为整个kernel只能配置一个BlockDim
    uint64_t randomWkSizeByte;              // 随机数生成部分需要的workspace大小
    uint32_t subNTileCount;                 // 随机数生成：切分N的块数
    uint32_t subNTile[SUB_N_TILE_COUNT];    // 随机数生成：N切分后，每块的元素数量
    uint32_t philoxKey[PHILOX_KEY_SIZE];
    uint32_t philoxOffset;
    int64_t n;
    struct SortRegBaseTilingDataForRandperm sortTilingData;
};

#endif