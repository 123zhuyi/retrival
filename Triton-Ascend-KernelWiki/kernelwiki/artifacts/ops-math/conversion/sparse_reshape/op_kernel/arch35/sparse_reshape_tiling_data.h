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
 * \file sparse_reshape_tiling_data.h
 * \brief Tiling data struct for sparse_reshape operator
 */

#ifndef SPARSE_RESHAPE_TILING_DATA_H_
#define SPARSE_RESHAPE_TILING_DATA_H_

constexpr int32_t MAX_RANK = 8;

struct SparseReshapeTilingData {
    int64_t nnz;                        // number of non-zero elements
    int32_t inputRank;                  // input tensor rank
    int32_t outputRank;                 // output tensor rank
    int32_t isIdentityReshape;          // 1 = input_shape == output_shape (fast path)
    int64_t inputStrides[MAX_RANK];     // input strides (precomputed on host)
    int64_t outputStrides[MAX_RANK];    // output strides (precomputed on host)
    int64_t outputShape[MAX_RANK];      // resolved output shape (-1 replaced)
};

#endif  // SPARSE_RESHAPE_TILING_DATA_H_
