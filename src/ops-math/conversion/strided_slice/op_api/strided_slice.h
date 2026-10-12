/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_STRIDED_SLICE_H
#define OP_API_INC_LEVEL0_STRIDED_SLICE_H

#include "opdev/op_def.h"

namespace l0op {

const aclTensor* StridedSlice(
    const aclTensor* x, const aclTensor* y, const aclTensor* begin, const aclTensor* end, const aclTensor* stride,
    aclOpExecutor* executor);

const aclTensor* StridedSlice(
    const aclTensor* self, const aclTensor* begin, const aclTensor* end, const aclTensor* strides, int64_t beginMask,
    int64_t endMask, int64_t ellipsisMask, int64_t newAxisMask, int64_t shrinkAxisMask, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_STRIDED_SLICE_H
