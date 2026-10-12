/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "logdet.h"

#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Logdet);

// 仅 fp32 走 AiCore
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore kernel: one input self, one output out.
static const aclTensor* LogdetAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(LogdetAiCore, self, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Logdet, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogdetAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

const aclTensor* Logdet(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Logdet, self);

    // 输出 batch 形状 = self.shape[:-2]
    auto shapeVec = ToShapeVector(self->GetViewShape());
    shapeVec.pop_back();
    shapeVec.pop_back();
    op::Shape targetShape;
    ToShape(shapeVec, targetShape);

    auto out = executor->AllocTensor(targetShape, self->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(out != nullptr, nullptr);

    INFER_SHAPE(Logdet, OP_INPUT(self), OP_OUTPUT(out));

    if (IsAiCoreSupport(self)) {
        return LogdetAiCore(self, out, executor);
    }
    return nullptr;
}
} // namespace l0op
