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
 * \file reduce_std_with_mean_tiling_data.h
 * \brief TilingData struct for Welford one-pass kernel
 *
 * Data layout: reduce-dim elements MUST be contiguous (reduce dims are
 * innermost after transpose in the L2 API layer). No srcStride needed.
 */

#ifndef _REDUCE_STD_WITH_MEAN_TILING_DATA_H_
#define _REDUCE_STD_WITH_MEAN_TILING_DATA_H_

struct ReduceStdWithMeanTilingData {
    int64_t totalNonReduce = 0; // total non-reduce elements across all cores
    int64_t reduceLength = 0;   // reduce dimension length
    int64_t blockFactor = 0;    // max non-reduce elements per core
    int64_t ubLength = 0;       // UB tile length for reduce dimension
    int64_t correction = 0;     // Bessel correction (0 or 1)
    float eps = 0.001f;         // numerical stability epsilon
    bool invert = false;        // if true, output 1/std instead of std
};

#endif // _REDUCE_STD_WITH_MEAN_TILING_DATA_H_
