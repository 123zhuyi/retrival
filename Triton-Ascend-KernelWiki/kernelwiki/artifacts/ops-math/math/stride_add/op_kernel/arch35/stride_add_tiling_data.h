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
 * \file stride_add_tiling_data.h
 * \brief StrideAdd tiling data struct
 */

#ifndef __STRIDE_ADD_TILING_DATA_H__
#define __STRIDE_ADD_TILING_DATA_H__

struct StrideAddTilingData {
    int64_t totalElements = 0;      // 输出总元素数 = N × c1_len × H × W × C0
    int64_t perCoreElements = 0;    // 单核处理元素数
    int64_t hwC0Size = 0;           // H × W × C0，用于坐标分解中 inner 部分的取余
    int64_t c1Len = 0;              // 输出 y 的 C1 维度长度（单位：C1 块数）
    int64_t x1NStride = 0;          // x1 的 N 维 stride = C1_x1 × H × W × C0
    int64_t x2NStride = 0;          // x2 的 N 维 stride = C1_x2 × H × W × C0
    int32_t x1C1Offset = 0;         // x1 在 C1 维度的偏移（单位：C1 块数），必须 ≥ 0
    int32_t x2C1Offset = 0;         // x2 在 C1 维度的偏移（单位：C1 块数），必须 ≥ 0
    int32_t needCoreNum = 0;        // 需要的核数
};

#endif // __STRIDE_ADD_TILING_DATA_H__