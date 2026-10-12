/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_GET_SHAPE_OP_KERNEL_ARCH35_GET_SHAPE_TILING_DATA_H_
#define OPS_GET_SHAPE_OP_KERNEL_ARCH35_GET_SHAPE_TILING_DATA_H_

#include <cstdint>

namespace GetShapeConst {
constexpr int64_t MAX_TOTAL_DIM = 128;
constexpr int64_t MAX_DIM_PER_TENSOR = 8;
constexpr int32_t BUFFER_NUM = 1;
} // namespace GetShapeConst

struct GetShapeTilingData {
    int32_t inputNum = 0;
};

#endif // OPS_GET_SHAPE_OP_KERNEL_ARCH35_GET_SHAPE_TILING_DATA_H_
