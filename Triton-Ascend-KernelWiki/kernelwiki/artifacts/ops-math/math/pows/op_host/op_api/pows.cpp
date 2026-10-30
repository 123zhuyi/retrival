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
 * \file pows.cpp
 * \brief
 */

#include "pows.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Pows);

// AICORE算子kernel
static const aclTensor* PowsAiCore(
    const aclTensor* self, const aclTensor* exponent, aclTensor* powsOut, aclOpExecutor* executor)
{
    L0_DFX(PowsAiCore, self, exponent, powsOut);

    ADD_TO_LAUNCHER_LIST_AICORE(Pows, OP_INPUT(self, exponent), OP_OUTPUT(powsOut));
    return powsOut;
}

const aclTensor* Pows(const aclTensor* self, const aclTensor* exponent, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), exponent->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(exponent->GetViewShape()).GetString());
        return nullptr;
    }

    auto powsOut = executor->AllocTensor(broadcastShape, self->GetDataType());
    CHECK_RET(powsOut != nullptr, nullptr);
    return PowsAiCore(self, exponent, powsOut, executor);
}
} // namespace l0op
