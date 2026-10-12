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
 * \file tensor_redirect_tiling_data.h
 * \brief tiling data struct for tensor_redirect
 */

#ifndef TENSOR_REDIRECT_TILING_DATA_H
#define TENSOR_REDIRECT_TILING_DATA_H

#include <cstdint>

// 标准 C++ POD struct
struct TensorRedirectTilingData {
    int64_t usedCoreNum = 0;           // 实际使用核数（SetBlockDim 用）
    int64_t blockFactor = 0;           // 单核循环次数
    int64_t tailBlockFactor = 0;       // 尾核循环次数
    int64_t ubFactor = 0;              // 单次循环元素数
    int64_t tailBlockTailUbFactor = 0; // 尾核尾循环元素数
};

#endif // TENSOR_REDIRECT_TILING_DATA_H
