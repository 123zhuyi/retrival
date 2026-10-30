/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stateless_uniform.h"
#include "op_api/aclnn_check.h"
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

OP_TYPE_REGISTER(StatelessUniform);

static const aclTensor* StatelessUniformAiCore(
    const aclTensor* inputSize, const aclTensor* seed, const aclTensor* offset,
    const aclTensor* from, const aclTensor* to, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StatelessUniformAiCore, inputSize, seed, offset, from, to);

    ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessUniform, OP_ATTR_NAMES({"dtype"}), OP_INPUT(inputSize, seed, offset, from, to),
        OP_OUTPUT(out), OP_ATTR(out->GetDataType()));
    return out;
}

const aclTensor* StatelessUniform(
    const aclTensor* self, uint64_t seed, uint64_t offset,
    double from, double to, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT64);

    // from/to 以 DT_DOUBLE 传入
    auto fromTensor = executor->ConvertToTensor(executor->AllocScalar(from), op::DataType::DT_DOUBLE);
    auto toTensor = executor->ConvertToTensor(executor->AllocScalar(to), op::DataType::DT_DOUBLE);

    auto seedTensor = executor->ConvertToTensor(executor->AllocScalar(seed), op::DataType::DT_INT64);
    auto offsetTensor = executor->ConvertToTensor(executor->AllocScalar(offset), op::DataType::DT_INT64);

    aclTensor* out = nullptr;
    if (self->GetDataType() == op::DataType::DT_FLOAT16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16, self->GetViewFormat());
    } else if (self->GetDataType() == op::DataType::DT_BF16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BF16, self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT, self->GetViewFormat());
    }

    return StatelessUniformAiCore(
        inputSize, seedTensor, offsetTensor, fromTensor, toTensor, out, executor);
}

const aclTensor* StatelessUniform(
    const aclTensor* self,
    const aclTensor* seedTensor, const aclTensor* offsetTensor,
    double from, double to, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT64);

    // from/to 以 DT_DOUBLE 传入
    auto fromTensor = executor->ConvertToTensor(executor->AllocScalar(from), op::DataType::DT_DOUBLE);
    auto toTensor = executor->ConvertToTensor(executor->AllocScalar(to), op::DataType::DT_DOUBLE);

    aclTensor* out = nullptr;
    if (self->GetDataType() == op::DataType::DT_FLOAT16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16, self->GetViewFormat());
    } else if (self->GetDataType() == op::DataType::DT_BF16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BF16, self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT, self->GetViewFormat());
    }

    return StatelessUniformAiCore(
        inputSize, seedTensor, offsetTensor, fromTensor, toTensor, out, executor);
}
} // namespace l0op
