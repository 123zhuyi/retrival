/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIAS_TILING_DATA_H_
#define BIAS_TILING_DATA_H_

#include <cstdint>

constexpr int64_t BIAS_MAX_DIM_SIZE = 8;

struct BiasTilingData {
    int64_t blockFormer = 0;
    int64_t ubFormer = 0;
    int64_t ubOuter = 0;
    int64_t ubTail = 0;
    int64_t blockTail = 0;
    int64_t shapeLen = 0;
    int64_t ubSplitAxis = 0;
    int64_t dimProductBeforeUbInner = 0;
    int64_t elemNum = 0;
    int64_t input0Dims[BIAS_MAX_DIM_SIZE] = {0};
    int64_t input1Dims[BIAS_MAX_DIM_SIZE] = {0};
    int64_t outputDims[BIAS_MAX_DIM_SIZE] = {0};
    int64_t input0Strides[BIAS_MAX_DIM_SIZE] = {0};
    int64_t input1Strides[BIAS_MAX_DIM_SIZE] = {0};
    int64_t outputStrides[BIAS_MAX_DIM_SIZE] = {0};
};

#endif // BIAS_TILING_DATA_H_
