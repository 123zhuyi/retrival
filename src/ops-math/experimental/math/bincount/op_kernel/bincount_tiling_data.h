/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bincount_tiling_data.h
 * \brief tiling data struct
 */

#ifndef BINCOUNT_TILING_DATA_H_
#define BINCOUNT_TILING_DATA_H_
struct BincountTilingData {
    uint64_t totalNum;    // 输入元素数 N
    uint64_t outLength;   // 输出长度 L
    uint64_t coreNum;     // 启用核数
    uint64_t bigCoreNum;  // 预留
    uint64_t tileDataNum; // 单次 CopyIn 元素数
    uint32_t hasWeights;  // 0/1
    uint32_t largeL;      // 0=UB 私有直方图快路径; 1=L 放不下 UB,走 GM 直接散射回退路径
};
#endif
