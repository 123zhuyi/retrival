/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_STATELESS_RANDOM_UNIFORM_V3_H
#define OP_API_STATELESS_RANDOM_UNIFORM_V3_H

#include "opdev/op_executor.h"

namespace l0op {

// 原标量接口（保留，向后兼容）
const aclTensor* StatelessRandomUniformV3(
    const aclTensor* self, uint64_t seed, uint64_t offset,
    float from, float to, int32_t v3KernelMode, aclOpExecutor* executor);

// 新增：tensor 接口（seed/offset 以 device tensor 形式传入）
const aclTensor* StatelessRandomUniformV3(
    const aclTensor* self,
    const aclTensor* seedTensor, const aclTensor* offsetTensor,
    float from, float to, int32_t v3KernelMode, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_STATELESS_RANDOM_UNIFORM_V3_H