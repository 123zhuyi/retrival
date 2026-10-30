/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
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

#ifndef __MASKED_FILL_TILING_DATA_H__
#define __MASKED_FILL_TILING_DATA_H__

struct MaskedFillTilingData {
    uint32_t xSize;
    uint32_t maskSize;
    uint32_t sShape[128];
    uint32_t maskShape[128];
    uint32_t xNumShapes;
    uint32_t maskNumShapes;

    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formerTileNum;
    uint32_t formerTileLength;
    uint32_t formerLastTileLength;
    uint32_t tailLength;
    uint32_t tailTileNum;
    uint32_t tailTileLength;
    uint32_t tailLastTileLength;

    void Init()
    {
        xSize = 0;
        maskSize = 0;
        xNumShapes = 0;
        maskNumShapes = 0;

        formerNum = 0;
        formerLength = 0;
        formerTileNum = 0;
        formerTileLength = 0;
        formerLastTileLength = 0;
        tailLength = 0;
        tailTileNum = 0;
        tailTileLength = 0;
        tailLastTileLength = 0;

        // 依次赋值
        int count = 128;
        for (int i = 0; i < count; i++) {
            sShape[i] = 0;
            maskShape[i] = 0;
        }
    }
};
#endif // __MASKED_FILL_TILING_DATA_H__