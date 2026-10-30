/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pack_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __PACK_V2_TILLING_DATA_H__
#define __PACK_V2_TILLING_DATA_H__

#include <cstdint>
const int32_t BLOCK_DIM = 8;
struct PackV2TilingData {
    // core / tile information
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t ubPartDataNum;
    uint64_t smallCoreTailDataNum;
    uint64_t bigCoreTailDataNum;
    uint64_t smallCoreLoopNum;
    uint64_t bigCoreLoopNum;
    uint64_t tailBlockNum;
    uint64_t inputDataNum;

    // tensor shape
    uint32_t totalLengthx;
    uint32_t totalLengthy;
    uint32_t totalLengthz;
    uint32_t x1;
    uint32_t x2;
    uint32_t y1;
    uint32_t y2;
    uint32_t z2;

    // tiling
    uint32_t small_tile_times;
    uint32_t big_tile_times;
    uint32_t small_tail_num;
    uint32_t big_tail_num;
    uint32_t big_core_num;
    uint32_t small_core_num;
    uint32_t small_tile_length;
    uint32_t big_tile_length;
    uint32_t tileNum;
    uint32_t core_tile_x1;

    // secondary tiling
    uint32_t ssmall_tile_times;
    uint32_t sbig_tile_times;
    uint32_t ssmall_tail_num;
    uint32_t sbig_tail_num;
    uint32_t sbig_core_num;
    uint32_t ssmall_core_num;
    uint32_t ssmall_tile_length;
    uint32_t sbig_tile_length;
    uint32_t core_tile_s1;

    // partition info
    uint32_t partnum;
    uint32_t partnumX;

    // select axis info
    int32_t startX[BLOCK_DIM];
    int32_t endX[BLOCK_DIM];
    int32_t rowsX[BLOCK_DIM];
    int32_t startY[BLOCK_DIM];
    int32_t endY[BLOCK_DIM];
    int32_t rowsY[BLOCK_DIM];

    int32_t d;
    int32_t dimNum;
};

#endif
