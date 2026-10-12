/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conversion/im2col/op_api/im2col.h"
#include <cstddef>
#include <limits>
#include <string>
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Im2col);

static const std::string PADDING_MODE = "CALCULATED";
static constexpr size_t DIM_N = 0;
static constexpr size_t DIM_C = 1;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 3;
static constexpr size_t PAIR_HEIGHT = 0;
static constexpr size_t PAIR_WIDTH = 1;
static constexpr size_t PAD_TOP = 0;
static constexpr size_t PAD_BOTTOM = 1;
static constexpr size_t PAD_LEFT = 2;
static constexpr size_t PAD_RIGHT = 3;

static bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool CalculateOutputDim(int64_t input, int64_t kernel, int64_t dilation, int64_t paddingBefore,
                               int64_t paddingAfter, int64_t stride, int64_t& output)
{
    if (stride <= 0) {
        return false;
    }
    const __int128 effectiveKernel = static_cast<__int128>(dilation) * (kernel - 1) + 1;
    const __int128 numerator = static_cast<__int128>(input) + paddingBefore + paddingAfter - effectiveKernel;
    if (numerator < 0) {
        return false;
    }
    const __int128 result = numerator / stride + 1;
    if (result <= 0 || result > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    output = static_cast<int64_t>(result);
    return true;
}

static bool Im2colInferShape(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                             const aclIntArray* padding, const aclIntArray* stride, op::Shape& outShape)
{
    if (self == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col self is null.");
        return false;
    }
    if (kernelSize == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col kernelSize is null.");
        return false;
    }
    if (dilation == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col dilation is null.");
        return false;
    }
    if (padding == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col padding is null.");
        return false;
    }
    if (stride == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col stride is null.");
        return false;
    }
    int64_t outH = 0;
    int64_t outW = 0;
    if (!CalculateOutputDim(self->GetViewShape().GetDim(DIM_H), (*kernelSize)[PAIR_HEIGHT], (*dilation)[PAIR_HEIGHT],
                            (*padding)[PAD_TOP], (*padding)[PAD_BOTTOM], (*stride)[PAIR_HEIGHT], outH)) {
        return false;
    }
    if (!CalculateOutputDim(self->GetViewShape().GetDim(DIM_W), (*kernelSize)[PAIR_WIDTH], (*dilation)[PAIR_WIDTH],
                            (*padding)[PAD_LEFT], (*padding)[PAD_RIGHT], (*stride)[PAIR_WIDTH], outW)) {
        return false;
    }
    int64_t outChannels = 0;
    if (!SafeMul(self->GetViewShape().GetDim(DIM_C), (*kernelSize)[PAIR_HEIGHT], outChannels) ||
        !SafeMul(outChannels, (*kernelSize)[PAIR_WIDTH], outChannels) || outChannels <= 0) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "Im2col output channel size is invalid.");
        return false;
    }
    int64_t outSpatial = 0;
    if (!SafeMul(outH, outW, outSpatial) || outSpatial <= 0) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "Im2col output spatial size is invalid.");
        return false;
    }
    outShape = {self->GetViewShape().GetDim(DIM_N), outChannels, outSpatial};
    return true;
}

const aclTensor* Im2col(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                        const aclIntArray* padding, const aclIntArray* stride, aclOpExecutor* executor)
{
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col executor is null.");
        return nullptr;
    }
    L0_DFX(Im2col, self, kernelSize, dilation, padding, stride);
    op::Shape outShape;
    if (!Im2colInferShape(self, kernelSize, dilation, padding, stride, outShape)) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "im2col infer shape failed.");
        return nullptr;
    }
    auto out = executor->AllocTensor(outShape, self->GetDataType(), self->GetViewFormat());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Im2col output tensor allocation failed.");
        return nullptr;
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Im2col, OP_INPUT(self), OP_OUTPUT(out),
                                           OP_ATTR(kernelSize, stride, dilation, PADDING_MODE, padding));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "Im2col ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op
