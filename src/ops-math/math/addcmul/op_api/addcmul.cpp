/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file addcmul.cpp
 * \brief
 */

#include "math/add/op_api/add.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Addcmul);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // Add只需要判断dtype
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* AddcmulAiCore(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* value,
    const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AddcmulAiCore, self, tensor1, tensor2, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Addcmul, OP_INPUT(self, tensor1, tensor2, value), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddcmulAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static const aclTensor* AddcmulAiCpu(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* value, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(AddcmulAiCpu, self, tensor1, tensor2, out);
    static internal::AicpuTaskSpace space("Addcmul");
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICPU(Addcmul, OP_ATTR_NAMES(), OP_INPUT(self, tensor1, tensor2, value), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddcmulAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

static bool InferShape(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, op::Shape& broadcastShape)
{
    if (!BroadcastInferShape(tensor1->GetViewShape(), tensor2->GetViewShape(), broadcastShape)) {
        return false;
    }
    if (!BroadcastInferShape(self->GetViewShape(), broadcastShape, broadcastShape)) {
        return false;
    }
    return true;
}

const aclTensor* Addcmul(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* value,
    aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!InferShape(self, tensor1, tensor2, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast failed.");
        return nullptr;
    }
    auto addcmulOut = executor->AllocTensor(broadcastShape, self->GetDataType());

    if (IsAiCoreSupport(self)) {
        return AddcmulAiCore(self, tensor1, tensor2, value, addcmulOut, executor);
    } else {
        return AddcmulAiCpu(self, tensor1, tensor2, value, addcmulOut, executor);
    }
}

} // namespace l0op