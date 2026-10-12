/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GCD_TILING_DATA_H_
#define GCD_TILING_DATA_H_

#include <cstdint>

constexpr int64_t GCD_MAX_DIMS = 8;

struct GcdTilingData {
    int64_t totalNum = 0;
    int64_t rank = 1;
    int64_t outputDims[GCD_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t x1Strides[GCD_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t x2Strides[GCD_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
};

#endif // GCD_TILING_DATA_H_
