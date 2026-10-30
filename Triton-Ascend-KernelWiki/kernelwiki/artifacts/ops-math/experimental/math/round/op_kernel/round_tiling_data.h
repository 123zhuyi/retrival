/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ROUND_TILING_DATA_H_
#define ROUND_TILING_DATA_H_
struct RoundTilingData {
    uint64_t smallCoreDataNum;  // 小核处理的总数据量大小
    uint64_t bigCoreDataNum;    // 大核处理的总数据量大小
    uint64_t finalBigTileNum;   // 大核上数据搬运次数
    uint64_t finalSmallTileNum; // 小核上数据搬运次数
    uint64_t tileDataNum;       // 单核单次可处理的数据量大小
    uint64_t smallTailDataNum;  // 小核最后一次搬运处理的数据量大小
    uint64_t bigTailDataNum;    // 大核最后一次搬运处理数据量大小
    uint64_t tailBlockNum;      // 大核个数
    int64_t decimals;
};
#endif
