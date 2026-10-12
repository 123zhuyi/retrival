/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsa_gen_bit_mask.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DSAGenBitMask);
OP_TYPE_REGISTER(DSAGenBitMaskTensor);

void DSAGenBitMask(
    uint64_t count, uint64_t seed, uint64_t offset, const aclScalar* dropout, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(DSAGenBitMask, count, seed, offset, dropout);
    ADD_TO_LAUNCHER_LIST_DSA(DSAGenBitMask, OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));
}

static const int64_t UINT8_BIT_NUMBER = 8;

const aclTensor* DSAGenBitMask(
    uint64_t count, uint64_t seed, uint64_t offset, const aclScalar* dropout, aclOpExecutor* executor)
{
    L0_DFX(DSAGenBitMask, count, seed, offset, dropout);

    auto out = executor->AllocTensor(op::Shape{static_cast<int64_t>(count) / UINT8_BIT_NUMBER}, op::DataType::DT_UINT8);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_DSA(DSAGenBitMask, OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));

    return out;
}

void DSAGenBitMaskTensor(
    uint64_t count, const aclTensor* seed, const aclTensor* offset, const aclScalar* dropout, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(DSAGenBitMaskTensor, count, seed, offset, dropout);
    ADD_TO_LAUNCHER_LIST_DSA(DSAGenBitMask, OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));
}

const aclTensor* DSAGenBitMaskTensor(
    uint64_t count, const aclTensor* seed, const aclTensor* offset, const aclScalar* dropout, aclOpExecutor* executor)
{
    L0_DFX(DSAGenBitMaskTensor, count, seed, offset, dropout);

    auto out = executor->AllocTensor(op::Shape{static_cast<int64_t>(count) / UINT8_BIT_NUMBER}, op::DataType::DT_UINT8);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_DSA(DSAGenBitMask, OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));

    return out;
}
} // namespace l0op
