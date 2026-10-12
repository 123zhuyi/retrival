/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi <@hitLeechi>
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
 * \file range.cpp
 * \brief
 */

#include "range.h"
template <uint32_t TYPE_START, uint32_t TYPE_END, uint32_t TYPE_STEP>
__global__ __aicore__ void range(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RangeTilingData);
    GET_TILING_DATA_WITH_STRUCT(RangeTilingData, tilingData, tiling);
    NsRange::Range<float, float, float> op;
    op.Init(
        start, end, step, z, tilingData.totalLength, tilingData.blockLength, tilingData.tileNum,
        tilingData.lastTileLength, tilingData.formerLength, tilingData.formerTileNum, tilingData.formerLastTileLength,
        tilingData.formerNum, tilingData.tailLength, tilingData.tailTileNum, tilingData.tailLastTileLength,
        tilingData.isEvenCore);
    op.Process();
}