/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file roll.cpp
 * @brief Roll L0 API implementation.
 */

#include "roll.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Roll);

aclTensor* RollAiCore(const aclTensor* x,
                      const aclIntArray* shifts,
                      const aclIntArray* dims,
                      aclTensor* rollOut,
                      aclOpExecutor* executor)
{
    (void)executor;
    L0_DFX(RollAiCore, x, shifts, dims);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Roll, OP_INPUT(x), OP_OUTPUT(rollOut), OP_ATTR(shifts, dims));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RollAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return rollOut;
}

const aclTensor* Roll(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclOpExecutor* executor)
{
    if (x == nullptr || executor == nullptr) {
        return nullptr;
    }
    auto rollOut = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    return Roll(x, shifts, dims, rollOut, executor);
}

const aclTensor* Roll(const aclTensor* x,
                      const aclIntArray* shifts,
                      const aclIntArray* dims,
                      aclTensor* out,
                      aclOpExecutor* executor)
{
    if (x == nullptr || out == nullptr || executor == nullptr) {
        return nullptr;
    }
    return RollAiCore(x, shifts, dims, out, executor);
}

} // namespace l0op
