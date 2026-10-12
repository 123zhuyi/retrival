/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_any.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ReduceAny);
const aclTensor* ReduceAny(const aclTensor* self, const aclIntArray* dim, bool keepDim, aclOpExecutor* executor)
{
    L0_DFX(ReduceAny, self, dim, keepDim);
    auto dims = executor->ConvertToTensor(dim, op::DataType::DT_INT64);
    const aclTensor* out = nullptr;
    bool isSpecialPlatform = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B || GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93;
    if (isSpecialPlatform && (self->GetDataType() == op::DataType::DT_BF16 || self->GetDataType() == op::DataType::DT_FLOAT)) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT);
    } else if (isSpecialPlatform && self->GetDataType() == op::DataType::DT_FLOAT16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16);
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BOOL);
    }
    // self为非scalar或者不需要保持原Tensor的size，需要reshape
    if (self->GetViewShape().GetDimNum() != 0 || !keepDim) {
        auto ret = INFER_SHAPE(ReduceAny, OP_INPUT(self, dims), OP_OUTPUT(out), OP_ATTR(keepDim));
        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
            return nullptr;
        }
    }
    auto retAicore =
        ADD_TO_LAUNCHER_LIST_AICORE(ReduceAny, op::AI_CORE, OP_INPUT(self, dims), OP_ATTR(keepDim), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ReduceAny ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op