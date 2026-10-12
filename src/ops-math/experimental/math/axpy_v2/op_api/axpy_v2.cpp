/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axpy_v2.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AxpyV2);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_UINT8, DataType::DT_INT8,  DataType::DT_INT64,   DataType::DT_BOOL,
};
// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// AICORE算子kernel
static const aclTensor* AxpyV2AiCore(
    const aclTensor* self, const aclTensor* other, const aclTensor* alpha, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(AxpyV2AiCore, self, other, alpha, out);
    if (!(self->GetDataType() == other->GetDataType() && self->GetDataType() == alpha->GetDataType() &&
          self->GetDataType() == out->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self, other, alpha, out dtype should all be same.");
        return nullptr;
    }
    // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将AxpyV2算子加入任务队列
    // op::OP_TYPE_AxpyV2是算子的OpType，self、other、 alpha是算子的输入，out是算子的输出
    ADD_TO_LAUNCHER_LIST_AICORE(AxpyV2, OP_INPUT(self, other, alpha), OP_OUTPUT(out));
    return out;
}

const aclTensor* AxpyV2(const aclTensor* self, const aclTensor* other, const aclTensor* alpha, aclOpExecutor* executor)
{
    // 检查输入参数
    L0_DFX(AxpyV2, self, other, alpha);
    auto out = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(AxpyV2, OP_INPUT(self, other, alpha), OP_OUTPUT(out), OP_EMPTY_ARG);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return nullptr;
    }
    if (!IsAiCoreSupport(self)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self dtype %s should be in dtype support list [%s].",
            op::ToString(self->GetDataType()).GetString(), op::ToString(AICORE_DTYPE_SUPPORT_LIST).GetString());
        return nullptr;
    }

    return AxpyV2AiCore(self, other, alpha, out, executor);
}
} // namespace l0op
