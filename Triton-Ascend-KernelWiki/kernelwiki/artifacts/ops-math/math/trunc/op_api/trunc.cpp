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
 * \file trunc.cpp
 * \brief
 */
#include "trunc.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Trunc);

static bool TruncInferShape(const op::Shape& selfShape, op::Shape& outShape)
{
    outShape = selfShape;
    return true;
}

const aclTensor* Trunc(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Trunc, self);

    op::Shape outShape;
    if (!TruncInferShape(self->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infer shape %s and %s failed.",
            op::ToString(self->GetStorageShape()).GetString(),
            op::ToString(outShape).GetString());
    return nullptr;
    }

    auto out = executor->AllocTensor(outShape, self->GetDataType());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Trunc, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

//Trunc内存地址复用
const aclTensor* InplaceTrunc(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(InplaceTrunc, self);

    op::Shape outShape;
    if (!TruncInferShape(self->GetViewShape(), outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infer shape %s and %s failed.",
            op::ToString(self->GetStorageShape()).GetString(),
            op::ToString(outShape).GetString());
    return nullptr;
    }

    auto out = const_cast<aclTensor*>(self);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Trunc, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

}  // namespace l0op
