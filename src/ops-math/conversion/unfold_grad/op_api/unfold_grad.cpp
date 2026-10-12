/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "unfold_grad.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/aicpu/aicpu_task.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UnfoldGrad);

static inline bool IsAiCoreSupport(
    const aclTensor* gradOut, int64_t dim, int64_t size, int64_t step)
{
    int64_t dimNum = gradOut->GetViewShape().GetDimNum()-1;
    int64_t maxv = size >= step ? size : step;
    op::DataType gradOutDtype = gradOut->GetDataType();
    if ((dim == dimNum - 1) &&
        (((gradOutDtype == DataType::DT_FLOAT) && (maxv <= 49088)) ||
         ((gradOutDtype == DataType::DT_FLOAT16 || gradOutDtype == DataType::DT_BF16) && (maxv <= 32720)))) {
            return true;
    } else if ((dim == dimNum - 2) &&
        (((gradOutDtype == DataType::DT_FLOAT) && (maxv <= 88)) ||
         ((gradOutDtype == DataType::DT_FLOAT16 || gradOutDtype == DataType::DT_BF16) && (maxv <= 72)))) {
            return true;
    } else {
        return false;
    }
}

// AICPU算子kernel
static const aclTensor* UnfoldGradAiCpu(
    const aclTensor* gradOut, const aclTensor* inputSizes, int64_t dim, int64_t size, int64_t step, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(UnfoldGradAiCpu, gradOut, inputSizes, dim, size, step, out);

    static internal::AicpuTaskSpace space("UnfoldGrad");
    ADD_TO_LAUNCHER_LIST_AICPU(
        UnfoldGrad, OP_ATTR_NAMES({"dim", "size", "step"}), OP_INPUT(gradOut, inputSizes), OP_OUTPUT(out),
        OP_ATTR(dim, size, step));
    return out;
}

// AICORE算子kernel
static const aclTensor* UnfoldGradAiCore(
    const aclTensor* gradOut, const aclTensor* inputSizes, int64_t dim, int64_t size, int64_t step, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(UnfoldGradAiCore, gradOut, inputSizes, dim, size, step, out);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE
    ADD_TO_LAUNCHER_LIST_AICORE(UnfoldGrad, OP_INPUT(gradOut, inputSizes), OP_OUTPUT(out), OP_ATTR(dim, size, step));
    return out;
}

const aclTensor* UnfoldGrad(
    const aclTensor* gradOut, const aclTensor* inputSizes, int64_t dim, int64_t size, int64_t step,
    aclOpExecutor* executor)
{
    L0_DFX(UnfoldGrad, gradOut, inputSizes, dim, size, step);
    auto out = executor->AllocTensor(gradOut->GetDataType(), gradOut->GetStorageFormat(), gradOut->GetOriginalFormat());
    auto ret = INFER_SHAPE(UnfoldGrad, OP_INPUT(gradOut, inputSizes), OP_OUTPUT(out), OP_ATTR(dim, size, step));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infershape failed.");
        return nullptr;
    }
    if (IsAiCoreSupport(gradOut, dim, size, step)) {
        return UnfoldGradAiCore(gradOut, inputSizes, dim, size, step, out, executor);
    } else {
        return UnfoldGradAiCpu(gradOut, inputSizes, dim, size, step, out, executor);
    }
}
} // namespace l0op
