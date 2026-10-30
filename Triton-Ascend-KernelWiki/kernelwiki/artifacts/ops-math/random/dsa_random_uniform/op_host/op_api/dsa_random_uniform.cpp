/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsa_random_uniform.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DSARandomUniform);
OP_TYPE_REGISTER(DSARandomUniformTensor);

const aclTensor* DSARandomUniform(
    const aclIntArray* outShape, uint64_t seed, uint64_t offset, const aclScalar* low, const aclScalar* high,
    aclOpExecutor* executor)
{
    L0_DFX(DSARandomUniform, outShape, seed, offset, low, high);
    op::Shape shape;
    op::ToShape(outShape->GetData(), outShape->Size(), shape);
    auto out = executor->AllocTensor(shape, low->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(out != nullptr, nullptr);
    uint64_t count = static_cast<uint64_t>(shape.GetShapeSize());
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomUniform, OP_INPUT(count, seed, offset, low, high), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}

const aclTensor* DSARandomUniformTensor(
    const aclIntArray* outShape, const aclTensor* seed, const aclTensor* offset, const aclScalar* low,
    const aclScalar* high, aclOpExecutor* executor)
{
    L0_DFX(DSARandomUniformTensor, outShape, seed, offset, low, high);
    op::Shape shape;
    op::ToShape(outShape->GetData(), outShape->Size(), shape);
    auto out = executor->AllocTensor(shape, low->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(out != nullptr, nullptr);
    uint64_t count = static_cast<uint64_t>(shape.GetShapeSize());
    ADD_TO_LAUNCHER_LIST_DSA(DSARandomUniform, OP_INPUT(count, seed, offset, low, high), OP_OUTPUT(out), OP_ATTR(0));
    return out;
}
} // namespace l0op
