/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mod.h"

#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Mod);

static const std::initializer_list<op::DataType> AICPU_DTYPE_SUPPORT_LIST = {op::DataType::DT_DOUBLE, op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_UINT8};

// AICore supports BF16, FP16, FP32 and INT32. Other README dtypes fall back to AICPU.
const aclTensor* ModAiCore(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ModAiCore, input, other, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Mod, OP_INPUT(input, other), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ModAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// TF-AICPU 支持FLOAT64
const aclTensor* ModAiCpu(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ModAiCpu, input, other, out);

    static internal::AicpuTaskSpace space("Mod", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Mod, OP_ATTR_NAMES(), OP_INPUT(input, other), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ModAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

// 只支持 AICORE
const aclTensor* Mod(const aclTensor* input, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(Mod, input, other);
    op::Shape broadcastShape;
    if (!BroadcastInferShape(input->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(input->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, input->GetDataType());

    if (CheckType(input->GetDataType(), AICPU_DTYPE_SUPPORT_LIST)) {
        return ModAiCpu(input, other, out, executor);
    } else {
        return ModAiCore(input, other, out, executor);
    }
}
} // namespace l0op