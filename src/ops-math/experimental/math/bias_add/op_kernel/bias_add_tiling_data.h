/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIAS_ADD_TILING_DATA_H_
#define BIAS_ADD_TILING_DATA_H_

#include <cstdint>

struct BiasAddTilingData {
    int64_t totalElements;
    int64_t channelSize;
    int64_t innerSize;
    int64_t smallCoreDataNum;
    int64_t bigCoreDataNum;
    int64_t finalBigTileNum;
    int64_t finalSmallTileNum;
    int64_t tileDataNum;
    int64_t biasCacheElems;
    int64_t brcTmpBytes;
    int64_t useFastPath;
    int64_t superCycleSize;
    int64_t kCycleCount;
    int64_t smallTailDataNum;
    int64_t bigTailDataNum;
    int64_t tailBlockNum;
};

#endif // BIAS_ADD_TILING_DATA_H_
