/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dropout_v3_grad.cpp
 * \brief
 */

#include "dropout_v3_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(DropOutV3Grad);

// AICORE算子kernel
static const aclTensor* DropoutV3GradAiCore(const aclTensor* gradY, const aclTensor* mask, const aclTensor* scale,
                                            const aclTensor* gradX, aclOpExecutor* executor)
{
    L0_DFX(DropoutV3GradAiCore, gradY, mask, scale, gradX);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DropOutV3Grad, OP_INPUT(gradY, mask, scale), OP_OUTPUT(gradX));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DropoutV3GradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return gradX;
}

const aclTensor* DropoutV3Grad(const aclTensor* gradY, const aclTensor* mask, const aclTensor* scale,
                               aclOpExecutor* executor)
{
    auto gradX = executor->AllocTensor(gradY->GetViewShape(), gradY->GetDataType());
    return DropoutV3GradAiCore(gradY, mask, scale, gradX, executor);
}
} // namespace l0op
