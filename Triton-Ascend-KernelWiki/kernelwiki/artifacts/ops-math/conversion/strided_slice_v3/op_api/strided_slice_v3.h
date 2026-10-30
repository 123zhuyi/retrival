/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_STRIDED_SLICE_V3_H_
#define OP_API_INC_LEVEL0_STRIDED_SLICE_V3_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* StridedSliceV3(
    const aclTensor* self, int64_t begin, int64_t end, int64_t axis, int64_t strideds, aclOpExecutor* executor);

const aclTensor* StridedSliceV3V2(
    const aclTensor* self, const aclIntArray* begins, const aclIntArray* ends, const aclIntArray* axis,
    const aclIntArray* strideds, aclOpExecutor* executor);
} // namespace l0op

#endif // #ifndef OP_API_INC_LEVEL0_STRIDED_SLICE_V3_H_
