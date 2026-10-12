/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_random.cpp
 * \brief Stateless random number generation API
 */

#include "stateless_random.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(StatelessRandom);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_BF16,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL,
    op::DataType::DT_INT64};

static inline bool IsDtypeSupport(DataType inputDtype)
{
    return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST);
}

static const aclTensor* StatelessRandomWithoutFromToAiCore(
    const aclTensor* inputSize, const aclTensor* seed, const aclTensor* offset,
    aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomWithoutFromToAiCore, inputSize, seed, offset);

    ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessRandom, OP_INPUT(inputSize, seed, offset, nullptr, nullptr),
        OP_OUTPUT(out), OP_ATTR(out->GetDataType()));
    return out;
}

const aclTensor* StatelessRandomWithoutFromTo(
    const aclTensor* self, int64_t seed, int64_t offset,
    aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT64);
    auto seedTensor = executor->ConvertToTensor(&seed, 1, op::DataType::DT_INT64);
    auto offsetTensor = executor->ConvertToTensor(&offset, 1, op::DataType::DT_INT64);

    aclTensor* out = nullptr;
    if (IsDtypeSupport(self->GetDataType())) {
        out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_INT32, self->GetViewFormat());
    }

    return StatelessRandomWithoutFromToAiCore(inputSize, seedTensor, offsetTensor, out, executor);
}

const aclTensor* StatelessRandomWithoutFromTo(
    const aclTensor* self, const aclTensor* seedTensor, const aclTensor* offsetTensor,
    aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT64);

    aclTensor* out = nullptr;
    if (IsDtypeSupport(self->GetDataType())) {
        out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_INT32, self->GetViewFormat());
    }

    return StatelessRandomWithoutFromToAiCore(inputSize, seedTensor, offsetTensor, out, executor);
}


static const aclTensor* StatelessRandomAiCore(
    const aclTensor* inputSize, const aclTensor* seed, const aclTensor* offset, const aclTensor* from,
    const aclTensor* to, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomAiCore, inputSize, seed, offset, from, to);

    ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessRandom, OP_INPUT(inputSize, seed, offset, from, to), OP_OUTPUT(out), OP_ATTR(out->GetDataType()));
    return out;
}

const aclTensor* StatelessRandom(
    const aclTensor* self, int64_t seed, int64_t offset, int64_t from, int64_t to,
    aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT64);
    auto seedTensor = executor->ConvertToTensor(&seed, 1, op::DataType::DT_INT64);
    auto offsetTensor = executor->ConvertToTensor(&offset, 1, op::DataType::DT_INT64);
    auto fromTensor = executor->ConvertToTensor(executor->AllocScalar(from), op::DataType::DT_INT64);
    auto toTensor = executor->ConvertToTensor(executor->AllocScalar(to), op::DataType::DT_INT64);

    aclTensor* out = nullptr;
    if (IsDtypeSupport(self->GetDataType())) {
        out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_INT32, self->GetViewFormat());
    }

    return StatelessRandomAiCore(inputSize, seedTensor, offsetTensor, fromTensor, toTensor, out, executor);
}

const aclTensor* StatelessRandom(
    const aclTensor* self, const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t from, int64_t to,
    aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT64);
    auto fromTensor = executor->ConvertToTensor(executor->AllocScalar(from), op::DataType::DT_INT64);
    auto toTensor = executor->ConvertToTensor(executor->AllocScalar(to), op::DataType::DT_INT64);

    aclTensor* out = nullptr;
    if (IsDtypeSupport(self->GetDataType())) {
        out = executor->AllocTensor(self->GetViewShape(), self->GetDataType(), self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_INT32, self->GetViewFormat());
    }

    return StatelessRandomAiCore(inputSize, seedTensor, offsetTensor, fromTensor, toTensor, out, executor);
}

} // namespace l0op