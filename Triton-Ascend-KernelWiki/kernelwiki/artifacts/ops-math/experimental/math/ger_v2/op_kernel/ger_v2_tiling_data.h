/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
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
 * \file ger_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

struct GerV2TilingData {
    int64_t smallCoreRows;   // 小核负责的行数
    int64_t bigCoreRows;     // 大核负责的行数
    int64_t tailRowNum;      // 尾行的数量，大核数量
    int64_t tileNums;        // 一行分为几个tile
    int64_t tileDataNum;     // 一次计算的数据个数
    int64_t tailTileDataNum; // 最后一次计算的数据个数
    int64_t m;               // 矩阵行数
    int64_t n;               // 矩阵列数
    float alpha;             // 参数
};
#endif