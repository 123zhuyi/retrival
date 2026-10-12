/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
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
 * \file strided_slice_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __STRIDED_SLICE_TILLING_DATA_H__
#define __STRIDED_SLICE_TILLING_DATA_H__

struct StridedSliceTilingData {
    uint32_t cols;
    uint32_t rows;
    uint32_t tileRows;
    uint32_t start1;
    uint32_t start2;
    uint32_t end1;
    uint32_t end2;
    uint32_t stride1;
    uint32_t stride2;
    uint32_t smallTailRows;
    uint32_t bigTailRows;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
    uint32_t baseRowsPerCore;
    uint32_t bigTotalRows;
    uint32_t tailRows;
};
#endif
