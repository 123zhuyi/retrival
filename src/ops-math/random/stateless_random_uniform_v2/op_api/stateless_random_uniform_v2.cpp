/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_random_uniform_v2.cpp
 * \brief
 */

#include "stateless_random_uniform_v2.h"
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

OP_TYPE_REGISTER(StatelessRandomUniformV2);
static const int64_t OFFSET_LIST_SIZE = 2;

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype)
{
    if (IsRegBase()) {
        return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// AICPU算子kernel 
static const aclTensor* StatelessRandomUniformV2AiCpu( 
    const aclTensor* inputSize, const aclTensor* seed, const aclTensor* offset, const aclTensor* alg, aclTensor* out, 
    aclOpExecutor* executor) 
{ 
    L0_DFX(StatelessRandomUniformV2AiCpu, inputSize, seed, offset, alg); 

    static internal::AicpuTaskSpace space("StatelessRandomUniformV2", ge::DEPEND_CONST_VALUE, true); 
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU( 
        StatelessRandomUniformV2, OP_ATTR_NAMES({"dtype", "Tshape"}), OP_INPUT(inputSize, seed, offset, alg), 
        OP_OUTPUT(out), OP_ATTR(out->GetDataType(), inputSize->GetDataType())); 
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr); 
    return out; 
}

// AICore算子kernel
static const aclTensor* StatelessRandomUniformV2AiCore(
    const aclTensor* inputSize, const aclTensor* seed, const aclTensor* offset, const aclTensor* alg, aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomUniformV2AiCore, inputSize, seed, offset, alg);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE, 将AiCore StatelessRandomUniformV2 算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessRandomUniformV2, OP_ATTR_NAMES({"dtype", "Tshape"}), OP_INPUT(inputSize, seed, offset, alg),
        OP_OUTPUT(out), OP_ATTR(out->GetDataType()));
    return out;
}

const aclTensor* StatelessRandomUniformV2(
    const aclTensor* self, uint64_t seed, uint64_t offset, int32_t alg, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT32);
    auto seedTensor = executor->ConvertToTensor(&seed, 1, op::DataType::DT_UINT64);
    FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
    aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), OFFSET_LIST_SIZE);
    auto offsetTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
    auto algTensor = executor->ConvertToTensor(executor->AllocScalar(alg), op::DataType::DT_INT32);
    aclTensor* out = nullptr;
    if (self->GetDataType() == op::DataType::DT_FLOAT16)
    {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16, self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT, self->GetViewFormat());
    }

    // aclnnInplaceRandom、aclnnInplaceUniform 接口，根据芯片类型判断AICore是否支持
    if (IsRegBase())
    {
        return StatelessRandomUniformV2AiCore(inputSize, seedTensor, offsetTensor, algTensor, out, executor);
    } else {
        return StatelessRandomUniformV2AiCpu(inputSize, seedTensor, offsetTensor, algTensor, out, executor);
    }

}

const aclTensor* StatelessRandomUniformV2(
    const aclTensor* self,
    const aclTensor* seedTensor, const aclTensor* offsetTensor,
    int32_t alg, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(self->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    auto inputSize = executor->ConvertToTensor(inputSizeArray, DataType::DT_INT32);

    auto algTensor = executor->ConvertToTensor(executor->AllocScalar(alg), op::DataType::DT_INT32);
    aclTensor* out = nullptr;
    if (self->GetDataType() == op::DataType::DT_FLOAT16)
    {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT16, self->GetViewFormat());
    } else {
        out = executor->AllocTensor(self->GetViewShape(), op::DataType::DT_FLOAT, self->GetViewFormat());
    }

    // 根据芯片类型、输入dtype判断AICore是否支持（out可能被映射为DT_FLOAT，需用原始输入dtype判断）
    if (IsAiCoreSupport(self->GetDataType()))
    {
        return StatelessRandomUniformV2AiCore(inputSize, seedTensor, offsetTensor, algTensor, out, executor);
    } else {
        return StatelessRandomUniformV2AiCpu(inputSize, seedTensor, offsetTensor, algTensor, out, executor);
    }
}
} // namespace l0op