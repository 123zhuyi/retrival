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
 * \file asinh_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _ASINH_TILING_DATA_H_
#define _ASINH_TILING_DATA_H_

struct AsinhTilingData {
    uint64_t formerCoreNum = 0;           // 前核数量
    uint64_t tailCoreNum = 0;             // 后核数量
    uint64_t formerCoreDataNum = 0;       // 前核分配总的数据量
    uint64_t tailCoreDataNum = 0;         // 后核分配总的数据量
    uint64_t formerCoreLoopCount = 0;     // 前核总共需要搬运次数
    uint64_t formerCoreFormerDataNum = 0; // 前核中每次搬运的数据量（除最后一次搬运）
    uint64_t formerCoreTailDataNum = 0;   // 前核中Uzi后一次搬运的数据量（整除情况下等于formerCoreFormerDataNum）
    uint64_t tailCoreLoopCount = 0;       // 后核总共需要搬运次数
    uint64_t tailCoreFormerDataNum = 0;   // 后核中每次搬运的数据量（除最后一次搬运）
    uint64_t tailCoreTailDataNum = 0;     // 后核中Uzi后一次搬运的数据量（整除情况下等于tailCoreFormerDataNum）
};
#endif