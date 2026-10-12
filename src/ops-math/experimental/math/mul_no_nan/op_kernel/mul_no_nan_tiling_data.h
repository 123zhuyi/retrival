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
 * \file mul_no_nan_tiling_data.h
 * \brief MulNoNan tiling data struct (shared by host tiling and device kernel)
 */
#ifndef MUL_NO_NAN_TILING_DATA_H_
#define MUL_NO_NAN_TILING_DATA_H_

#include <cstdint>

constexpr int64_t MUL_NO_NAN_MAX_DIMS = 8;

struct MulNoNanTilingData {
    int64_t totalNum = 0;
    int64_t rank = 1;
    int64_t outputDims[MUL_NO_NAN_MAX_DIMS] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t x1Strides[MUL_NO_NAN_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t x2Strides[MUL_NO_NAN_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
};

#endif // MUL_NO_NAN_TILING_DATA_H_
