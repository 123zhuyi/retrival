/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "masked_select.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(MaskedSelect);
OP_TYPE_REGISTER(MaskedSelectV3);

// AICORE算子kernel
static const aclTensor *MaskedSelectAiCore(const aclTensor *self, const aclTensor *mask, aclTensor *out, aclOpExecutor *executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetStorageShape(), mask->GetStorageShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(mask->GetViewShape()).GetString());
        return nullptr;
    }

    L0_DFX(MaskedSelectAiCore, self, mask, out);
    // outShapeTensor用于执行期放置实际shape，并用于刷新out的大小。
    Shape outShapeShape{2};
    auto outShapeTensor = executor->AllocTensor(outShapeShape, DataType::DT_INT64, Format::FORMAT_ND);
    CHECK_RET(outShapeTensor != nullptr, nullptr);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将MaskedSelect算子加入任务队列
    // MaskedSelect是算子的OpType，self是算子的输入，out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MaskedSelectV3, OP_INPUT(self, mask), OP_OUTPUT(out), OP_OUTSHAPE({outShapeTensor, 0}));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedSelectV3AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

// AICPU算子kernel
static const aclTensor *MaskedSelectAiCpu(const aclTensor *self, const aclTensor *mask, aclTensor *out, aclOpExecutor *executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetStorageShape(), mask->GetStorageShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(mask->GetViewShape()).GetString());
        return nullptr;
    }

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu MaskedSelect算子加入任务队列
    // MaskedSelect是算子的OpType，self、mask是算子的输入，maskedSelectOut是算子的输出
    L0_DFX(MaskedSelectAiCpu, self, mask, out);
    static internal::AicpuTaskSpace space("MaskedSelect", ge::DEPEND_SHAPE_RANGE);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(MaskedSelect, OP_ATTR_NAMES(), OP_INPUT(self, mask), OP_OUTPUT(out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* MaskedSelect(const aclTensor* self, const aclTensor* mask, aclTensor* out, aclOpExecutor* executor)
{
    return MaskedSelectAiCpu(self, mask, out, executor);
}
const aclTensor* MaskedSelectV3(const aclTensor* self, const aclTensor* mask, aclTensor* out, aclOpExecutor* executor)
{
    return MaskedSelectAiCore(self, mask, out, executor);
}
} // namespace l0op