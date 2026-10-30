/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh_grad_tiling_data.h
 * \brief tiling data struct
 */
#ifndef TANH_GRAD_TILING_DATA_H_
#define TANH_GRAD_TILING_DATA_H_
struct TanhGradTilingData {
    uint64_t smallCoreDataNum;  // 小核（小任务）处理的数据量
    uint64_t bigCoreDataNum;    // 大核（大任务）处理的数据量
    uint64_t finalBigTileNum;   // 大核的 tile 数量
    uint64_t finalSmallTileNum; // 小核的 tile 数量
    uint64_t tileDataNum;       // 每个 tile 的数据量
    uint64_t smallTailDataNum;  // 小核尾部剩余数据量
    uint64_t bigTailDataNum;    // 大核尾部剩余数据量
    uint64_t tailBlockNum;      // 处理尾部数据的核心数
};
#endif