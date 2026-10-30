/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_VIEWCOPY_H_
#define OP_API_INC_LEVEL0_VIEWCOPY_H_

#include <cstdint>

#include "opdev/common_types.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* ViewCopy(const aclTensor* dst, const aclTensor* dstSize, const aclTensor* dstStride,
                          const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                          const aclTensor* srcStride, const aclTensor* srcStorageOffset, const aclTensor* dstOut,
                          aclOpExecutor* executor);

const aclTensor* ViewCopy(const aclTensor* dst, const op::Shape& dstSize, const op::Strides& dstStride,
                          int64_t dstStorageOffset, const aclTensor* src, const op::Shape& srcSize,
                          const op::Strides& srcStride, int64_t srcStorageOffset, const aclTensor* dstOut,
                          aclOpExecutor* executor);

const aclTensor* ViewCopy(const aclTensor* dstOut, const aclTensor* dstSize, const aclTensor* dstStride,
                          const aclTensor* dstStorageOffset, const aclTensor* src, const aclTensor* srcSize,
                          const aclTensor* srcStride, const aclTensor* srcStorageOffset, aclOpExecutor* executor);

const aclTensor* ViewCopy(const aclTensor* dstOut, const op::Shape& dstSize, const op::Strides& dstStride,
                          int64_t dstStorageOffset, const aclTensor* src, const op::Shape& srcSize,
                          const op::Strides& srcStride, int64_t srcStorageOffset, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_VIEWCOPY_H_
