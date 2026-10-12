/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file concat_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __CONCAT_V2_TILLING_DATA_H__
#define __CONCAT_V2_TILLING_DATA_H__

#include <cstdint>

struct ConcatV2TilingData {
    // core / tile information
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;

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
    int32_t startX[8];
    int32_t endX[8];
    int32_t rowsX[8];
    int32_t startY[8];
    int32_t endY[8];
    int32_t rowsY[8];

    int32_t d;
    int32_t dimNum;
};

#endif
