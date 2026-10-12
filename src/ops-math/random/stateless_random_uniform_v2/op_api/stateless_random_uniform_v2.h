/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_UNIFORM_H_
#define OP_API_INC_LEVEL0_UNIFORM_H_

#include "opdev/op_executor.h"

namespace l0op {
// 原标量接口（保留，向后兼容）
const aclTensor* StatelessRandomUniformV2(
    const aclTensor* self, uint64_t seed, uint64_t offset, int32_t alg, aclOpExecutor* executor);

// 新增：tensor 接口（seed/offset 以 device tensor 形式传入，不做 D2H 拷贝）
const aclTensor* StatelessRandomUniformV2(
    const aclTensor* self,
    const aclTensor* seedTensor, const aclTensor* offsetTensor,
    int32_t alg, aclOpExecutor* executor);
}

#endif // OP_API_INC_LEVEL0_UNIFORM_H_