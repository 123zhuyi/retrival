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
 * \file div_v3.cpp
 * \brief DivV3 L0 API implementation
 */

#include "div_v3.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DivV3);

static const aclTensor* DivV3AiCore(
    const aclTensor* self, const aclTensor* other, int64_t mode, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(DivV3AiCore, self, other, out);

    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(DivV3, OP_INPUT(self), OP_INPUT(other), OP_OUTPUT(out), OP_ATTR("mode", mode));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DivV3AiCore add to launcher failed."), return nullptr);
    return out;
}

const aclTensor* DivV3(const aclTensor* self, const aclTensor* other, int64_t mode, aclOpExecutor* executor)
{
    Shape outShape = self->GetViewShape();
    const aclTensor* out = executor->AllocTensor(outShape, self->GetDataType());
    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DivV3 alloc output tensor failed."), return nullptr);

    return DivV3AiCore(self, other, mode, out, executor);
}

} // namespace l0op
