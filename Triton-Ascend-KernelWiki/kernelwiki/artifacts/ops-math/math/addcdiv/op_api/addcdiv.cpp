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
 * \file addcdiv.cpp
 * \brief
 */

#include "addcdiv.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Addcdiv);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor* self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool is910BSocVersion =
        (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
         IsRegBase());
    const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST =
        is910BSocVersion ? AICORE_DTYPE_SUPPORT_LIST_910B : AICORE_DTYPE_SUPPORT_LIST_910;
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool CanBroadcast(const aclTensor* t1, const aclTensor* t2, const aclTensor* t3, op::Shape& broadcastShape)
{
    op::Shape tempShape;
    if (BroadcastInferShape(t1->GetViewShape(), t2->GetViewShape(), tempShape)) {
        return BroadcastInferShape(t3->GetViewShape(), tempShape, broadcastShape);
    }
    return false;
}

static const aclTensor* AddcdivAiCore(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* value,
    const aclTensor* addcdivOut, aclOpExecutor* executor)
{
    L0_DFX(AddcdivAiCore, self, tensor1, tensor2, value);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Addcdiv, OP_INPUT(self, tensor1, tensor2, value), OP_OUTPUT(addcdivOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddcdivAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return addcdivOut;
}

static const aclTensor* AddcdivAiCpu(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* value,
    const aclTensor* addcdivOut, aclOpExecutor* executor)
{
    L0_DFX(AddcdivAiCpu, self, tensor1, tensor2, value);

    static internal::AicpuTaskSpace space("Addcdiv", ge::DEPEND_IN_SHAPE, false);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Addcdiv, OP_ATTR_NAMES(), OP_INPUT(self, tensor1, tensor2, value), OP_OUTPUT(addcdivOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddcdivAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return addcdivOut;
}

const aclTensor* Addcdiv(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclTensor* value,
    aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!CanBroadcast(tensor1, tensor2, self, broadcastShape)) {
        OP_LOGE(
            ACL_ERROR_INVALID_PARAM, "broadcast self: %s, tensor1: %s, tensor2: %s failed.",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(tensor1->GetViewShape()).GetString(),
            op::ToString(tensor2->GetViewShape()).GetString());
        return nullptr;
    }
    auto addcdivOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return AddcdivAiCore(self, tensor1, tensor2, value, addcdivOut, executor);
    } else {
        return AddcdivAiCpu(self, tensor1, tensor2, value, addcdivOut, executor);
    }
}
} // namespace l0op
