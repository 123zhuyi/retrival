/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "diagv2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

namespace l0op {
OP_TYPE_REGISTER(DiagV2);

const aclTensor* DiagV2(const aclTensor* self, int64_t diagonal, aclOpExecutor* executor)
{
    L0_DFX(DiagV2, self);
    // 固定写法，创建OpExecutor
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    auto ret = INFER_SHAPE(DiagV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(diagonal));
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(DiagV2, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(diagonal));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "DiagV2 add to aicore launch list failed.");
    return out;
}
} // namespace l0op
