/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_mat_mat_elements_plus.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AddMatMatElementsPlus);

static const aclTensor* AddMatMatElementsPlusAiCore(const aclTensor* c, const aclTensor* a, const aclTensor* b,
                                                    const aclTensor* beta, const aclTensor* alpha, const aclTensor* out,
                                                    aclOpExecutor* executor)
{
    L0_DFX(AddMatMatElementsPlusAiCore, c, a, b, beta, alpha, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AddMatMatElementsPlus, OP_INPUT(c, a, b, beta, alpha), OP_OUTPUT(out));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddMatMatElementsPlusAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

static bool InferBroadcastShape(const aclTensor* c, const aclTensor* a, const aclTensor* b, op::Shape& broadcastShape)
{
    if (!BroadcastInferShape(a->GetViewShape(), b->GetViewShape(), broadcastShape)) {
        return false;
    }
    if (!BroadcastInferShape(c->GetViewShape(), broadcastShape, broadcastShape)) {
        return false;
    }
    return true;
}

const aclTensor* AddMatMatElementsPlus(const aclTensor* c, const aclTensor* a, const aclTensor* b,
                                       const aclTensor* beta, const aclTensor* alpha, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!InferBroadcastShape(c, a, b, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast failed.");
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, c->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed for output.");
        return nullptr;
    }
    return AddMatMatElementsPlusAiCore(c, a, b, beta, alpha, out, executor);
}

} // namespace l0op
