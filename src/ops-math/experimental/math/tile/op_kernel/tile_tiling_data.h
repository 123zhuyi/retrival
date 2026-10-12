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
 * \file tile_tiling_data.h
 * \brief tiling data struct
 */
#ifndef TILE_TILING_DATA_H_
#define TILE_TILING_DATA_H_

#include <cstdint>

constexpr int32_t TILE_MAX_DIM = 8;

struct TileTilingData {
    int32_t numDims;
    int32_t inputShape[TILE_MAX_DIM];
    int32_t multiples[TILE_MAX_DIM];
    int32_t outputShape[TILE_MAX_DIM];
    int32_t inputStrides[TILE_MAX_DIM];
    int32_t outputStrides[TILE_MAX_DIM];
    int32_t totalInputElems;
    int32_t totalOutputElems;
    int32_t elemBytes;
    int32_t blockDim;
    int32_t ubSize;
    int32_t repeatPeriod;
    int32_t repeatInputPeriod;
    int32_t periodsPerSource;
    int32_t nUniqueSources;
};

#endif
