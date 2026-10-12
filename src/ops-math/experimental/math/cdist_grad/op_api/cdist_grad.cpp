/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "cdist_grad.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
static const int64_t NUMBER_TWO = 2;

OP_TYPE_REGISTER(CdistGrad);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static inline bool IsAiCoreSupport(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist)
{
    if (grad->GetDataType() != x1->GetDataType() || grad->GetDataType() != x2->GetDataType() ||
        grad->GetDataType() != cdist->GetDataType()) {
        return false;
    }
    return op::CheckType(grad->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           op::CheckType(x1->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           op::CheckType(x2->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
           op::CheckType(cdist->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor* CdistGradAiCore(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, float p, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(CdistGradAiCore, x1, x2, out);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore CdistGrad算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CdistGrad, OP_INPUT(grad, x1, x2, cdist), OP_OUTPUT(out), OP_ATTR(p));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CdistGradAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* CdistGrad(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, float p,
    aclOpExecutor* executor)
{
    op::Shape outputShape;
    auto dimnum = grad->GetViewShape().GetDimNum();
    for (size_t i = 0; i < dimnum - NUMBER_TWO; i++) {
        outputShape.AppendDim(grad->GetViewShape().GetDim(i));
    }
    outputShape.AppendDim(grad->GetViewShape().GetDim(dimnum - 1));
    // 根据输出shape申请输出tensor
    auto out = executor->AllocTensor(outputShape, grad->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }
    if (IsAiCoreSupport(grad, x1, x2, cdist)) {
        return CdistGradAiCore(grad, x1, x2, cdist, p, out, executor);
    } else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Datatype not supported.");
        return nullptr;
    }
}
} // namespace l0op