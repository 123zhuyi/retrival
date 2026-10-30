/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "im2col.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Im2col);

static const string PADDING_MODE = "CALCULATED";

static bool Im2colInferShape(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, op::Shape& outShape)
{
    int64_t outH =
        (self->GetViewShape().GetDim(2) + 2 * (*padding)[0] - ((*dilation)[0] * ((*kernelSize)[0] - 1) + 1)) /
            (*stride)[0] +
        1;
    int64_t outW =
        (self->GetViewShape().GetDim(3) + 2 * (*padding)[2] - ((*dilation)[1] * ((*kernelSize)[1] - 1) + 1)) /
            (*stride)[1] +
        1;
    outShape = {
        self->GetViewShape().GetDim(0), self->GetViewShape().GetDim(1) * (*kernelSize)[0] * (*kernelSize)[1],
        outH * outW};
    return true;
}

const aclTensor* Im2col(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, aclOpExecutor* executor)
{
    L0_DFX(Im2col, self, kernelSize, dilation, padding, stride);
    op::Shape outShape;
    if (!Im2colInferShape(self, kernelSize, dilation, padding, stride, outShape)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "im2col infer shape failed.");
        return nullptr;
    }
    auto out = executor->AllocTensor(outShape, self->GetDataType(), self->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Im2col, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(kernelSize, stride, dilation, PADDING_MODE, padding));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "Im2col ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op
