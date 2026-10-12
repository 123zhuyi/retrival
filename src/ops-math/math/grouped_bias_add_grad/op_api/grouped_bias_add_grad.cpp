/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_bias_add_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

OP_TYPE_REGISTER(GroupedBiasAddGrad);

const aclTensor* GroupedBiasAddGrad(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, int64_t groupIdxType, aclOpExecutor* executor)
{
    L0_DFX(GroupedBiasAddGrad, gradY, groupIdxOptional, groupIdxType);

    op::Shape outShape;
    if (groupIdxOptional != nullptr) {
        outShape.AppendDim(groupIdxOptional->GetViewShape().GetDim(0));
        outShape.AppendDim(gradY->GetViewShape().GetDim(GRAD_Y_SHAPE_WITH_GROUP_IDX - 1));
    } else {
        outShape.AppendDim(gradY->GetViewShape().GetDim(0));
        outShape.AppendDim(gradY->GetViewShape().GetDim(GRAD_Y_SHAPE_NO_GROUP_IDX - 1));
    }

    // 根据输入shape申请输出tensor
    auto gradBias = executor->AllocTensor(outShape, gradY->GetDataType(), op::Format::FORMAT_ND);
    if (gradBias == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc gradBias tensor failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        GroupedBiasAddGrad, OP_INPUT(gradY, groupIdxOptional), OP_OUTPUT(gradBias), OP_ATTR(groupIdxType));

    return gradBias;
}

} // namespace l0op
