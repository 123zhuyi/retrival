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
 * \file cross_struct.h
 * \brief cross tiling data
 */

#ifndef CROSS_STRUCT_H
#define CROSS_STRUCT_H

#include <cstdint>

constexpr int64_t MAX_DIM = 8;

#pragma pack(push, 8)
struct CrossRegbaseTilingData {
    int64_t totalVectors;
    int64_t vectorsPerCore;
    int64_t coreNum;
    int64_t dim;
    int64_t dimNum;
    int64_t mergedStride[MAX_DIM];
    int64_t x1Stride[MAX_DIM];
    int64_t x2Stride[MAX_DIM];
    int64_t yStride[MAX_DIM];
    int64_t dimStride;
    int64_t formerCore;
    int64_t usedInt64;
};
#pragma pack(pop)

#endif
