/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stateless_random_uniform_v3.h"
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

OP_TYPE_REGISTER(StatelessRandomUniformV3);
static const int64_t OFFSET_LIST_SIZE = 2;

static const aclTensor* StatelessRandomUniformV3AiCore(
    const aclTensor* inputSize, const aclTensor* seed, const aclTensor* offset,
    const aclTensor* from, const aclTensor* to, int32_t v3KernelMode, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomUniformV3AiCore, inputSize, seed, offset, from, to);

    ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessRandomUniformV3, OP_ATTR_NAMES({"dtype", "v3KernelMode"}), OP_INPUT(inputSize, seed, offset, from, to),
        OP_OUTPUT(out), OP_ATTR(out->GetDataType(), v3KernelMode));
    return out;
}

const aclTensor* StatelessRandomUniformV3(
    const aclTensor* self, uint64_t seed, uint64_t offset,
    float from, float to, int32_t v3KernelMode, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT32);
    auto seedTensor = executor->ConvertToTensor(&seed, 1, op::DataType::DT_UINT64);
    FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
    aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), OFFSET_LIST_SIZE);
    auto offsetTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
    auto fromTensor = executor->ConvertToTensor(executor->AllocScalar(from), op::DataType::DT_FLOAT);
    auto toTensor = executor->ConvertToTensor(executor->AllocScalar(to), op::DataType::DT_FLOAT);

    aclTensor* out = nullptr;
    if (self->GetDataType() == op::DataType::DT_FLOAT16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16, self->GetViewFormat());
    } else if (self->GetDataType() == op::DataType::DT_BF16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BF16, self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT, self->GetViewFormat());
    }

    return StatelessRandomUniformV3AiCore(
        inputSize, seedTensor, offsetTensor, fromTensor, toTensor, v3KernelMode, out, executor);
}

const aclTensor* StatelessRandomUniformV3(
    const aclTensor* self,
    const aclTensor* seedTensor, const aclTensor* offsetTensor,
    float from, float to, int32_t v3KernelMode, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT32);

    auto fromTensor = executor->ConvertToTensor(executor->AllocScalar(from), op::DataType::DT_FLOAT);
    auto toTensor = executor->ConvertToTensor(executor->AllocScalar(to), op::DataType::DT_FLOAT);

    aclTensor* out = nullptr;
    if (self->GetDataType() == op::DataType::DT_FLOAT16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16, self->GetViewFormat());
    } else if (self->GetDataType() == op::DataType::DT_BF16) {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_BF16, self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT, self->GetViewFormat());
    }

    return StatelessRandomUniformV3AiCore(
        inputSize, seedTensor, offsetTensor, fromTensor, toTensor, v3KernelMode, out, executor);
}
} // namespace l0op