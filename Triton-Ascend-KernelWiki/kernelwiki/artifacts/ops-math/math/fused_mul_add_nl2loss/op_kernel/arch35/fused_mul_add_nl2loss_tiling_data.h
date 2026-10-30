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
 * \file fused_mul_add_nl2loss_tiling_data.h
 * \brief tiling data struct shared by host tiling and arch35 kernel
 *
 * y1 = x1 * x3 + x2   (elementwise, x3 标量广播)
 * y2 = Σ(x1² / 2)     (全量 reduce, 标量输出)
 */

#ifndef FUSED_MUL_ADD_NL2LOSS_TILING_DATA_H
#define FUSED_MUL_ADD_NL2LOSS_TILING_DATA_H

#include <cstdint>

struct FusedMulAddNL2lossTilingData {
    int64_t totalElements; // N = prod(shape)，展平后总元素数
    int64_t coreElements; // 前 usedCores-1 个核每核处理的元素数（= totalElements / usedCores，不保证 VL 对齐）
    int64_t tailCoreElements; // 最后一个核处理的元素数
    int64_t ubTileSize;       // 每次 UB tile 处理的元素数（64 对齐）
};

#endif // FUSED_MUL_ADD_NL2LOSS_TILING_DATA_H
