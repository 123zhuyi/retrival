/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_DSA_GEN_BIT_MASK_H_
#define OP_API_INC_LEVEL0_DSA_GEN_BIT_MASK_H_

#include "opdev/op_executor.h"

namespace l0op {
void DSAGenBitMask(
    uint64_t count, uint64_t seed, uint64_t offset, const aclScalar* dropout, aclTensor* out, aclOpExecutor* executor);

const aclTensor* DSAGenBitMask(
    uint64_t count, uint64_t seed, uint64_t offset, const aclScalar* dropout, aclOpExecutor* executor);

void DSAGenBitMaskTensor(
    uint64_t count, const aclTensor* seed, const aclTensor* offset, const aclScalar* dropout, aclTensor* out,
    aclOpExecutor* executor);

const aclTensor* DSAGenBitMaskTensor(
    uint64_t count, const aclTensor* seed, const aclTensor* offset, const aclScalar* dropout, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_DSA_GEN_BIT_MASK_H_
