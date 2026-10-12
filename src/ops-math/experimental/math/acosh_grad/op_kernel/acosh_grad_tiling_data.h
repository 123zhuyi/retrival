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
 * \file acosh_grad_tiling_data.h
 * \brief tiling data struct for AcoshGrad
 */

#ifndef _ACOSHGRAD_TILING_DATA_H_
#define _ACOSHGRAD_TILING_DATA_H_

#include <cstdint>

struct AcoshGradTilingData {
    uint64_t totalLength;        // 展平后的元素总数
    uint64_t blockLength;        // 普通 core 处理的元素数
    uint64_t tailBlockLength;    // 最后一个 core（尾 core）处理的元素数
    uint64_t tileLength;         // 每次 tile 处理的元素数（UB 粒度，已对齐）
    uint64_t tileNum;            // 普通 core 的 tile 数
    uint64_t lastTileLength;     // 普通 core 最后一个 tile 的真实元素数
    uint64_t tailTileNum;        // 尾 core 的 tile 数
    uint64_t tailLastTileLength; // 尾 core 最后一个 tile 的真实元素数

    uint32_t coreNum;       // 实际启动 core 数
    uint32_t formerCoreNum; // blockLength > tailBlockLength 的 core 数（前段核）
};
#endif // _ACOSHGRAD_TILING_DATA_H_
