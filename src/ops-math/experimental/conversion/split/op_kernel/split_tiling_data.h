/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group Harbin Institute of Technology (HIT).
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
 * \file split_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __SPLIT_TILLING_DATA_H__
#define __SPLIT_TILLING_DATA_H__

struct SplitTilingData {
    uint32_t smallCoreDataNum{0};
    uint32_t bigCoreDataNum{0};
    uint32_t finalBigTileNum{0};
    uint32_t finalSmallTileNum{0};
    uint32_t tileDataNum{0};
    uint32_t smallTailDataNum{0};
    uint32_t bigTailDataNum{0};
    uint32_t tailBlockNum{0};
    uint32_t blockSize{0};
    // ===== Split 语义 =====
    int64_t axis{0};
    bool isEven{0};           // 判断是否是均分
    uint32_t indices_len{0};  // 切分数量
    uint32_t splitLen[11]{0}; // 切片长度
    // ===== Shape / 参数 =====
    uint32_t shape[8]{0};                // 输入张量的形状信息
    uint32_t indices_or_sections[10]{0}; // 支持数组形式的切分参数
    // ===== 辅助 =====
    uint32_t unit{0};
    uint32_t totalNums{0};
    uint32_t srcdim{0};
};
#endif