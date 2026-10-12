/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND EXPRESS OR IMPLIED
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file expandv_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __EXPANDV_TILLING_DATA_H__
#define __EXPANDV_TILLING_DATA_H__

struct ExpandvTilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t finalBigTileNum;
    uint64_t finalSmallTileNum;
    uint64_t tileDataNum;
    uint64_t smallTailDataNum;
    uint64_t bigTailDataNum;
    uint64_t tailBlockNum;

    // shape and stride info
    uint64_t in_rank;
    uint64_t out_rank;
    uint64_t inShapeArr[10];
    uint64_t outShapeArr[10];
    uint64_t inStrideArr[10];
    uint64_t outStrideArr[10];
};
#endif
