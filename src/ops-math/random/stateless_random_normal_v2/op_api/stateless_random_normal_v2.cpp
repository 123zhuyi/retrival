/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stateless_random_normal_v2.h"
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

OP_TYPE_REGISTER(StatelessRandomNormalV2);

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

// AICORE算子kernel
static inline const aclTensor* StatelessRandomNormalV2AiCore(
    const aclTensor* result, const aclTensor* shapeTensor, const aclTensor* keyTensor, const aclTensor* counterTensor,
    const aclTensor* algTensor, aclTensor* outTensor, aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomNormalV2AiCore, shapeTensor, keyTensor, counterTensor, algTensor, outTensor);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore StatelessRandomNormalV2算子加入任务队列
    auto dtype = result->GetDataType();
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessRandomNormalV2, OP_INPUT(shapeTensor, keyTensor, counterTensor, algTensor), OP_OUTPUT(outTensor),
        OP_ATTR(dtype));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return outTensor;
}

// **************************************************************************************************
// ************************* [AI CPU] StatelessRandomNormalV2算子调用入口  *****************************
// **************************************************************************************************

// AI CPU算子kernel
static const aclTensor* StatelessRandomNormalV2AiCpu(
    const aclTensor* result, const aclTensor* shapeTensor, const aclTensor* keyTensor, const aclTensor* counterTensor,
    const aclTensor* algTensor, aclTensor* outTensor, aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomNormalV2AiCpu, shapeTensor, keyTensor, counterTensor, algTensor, outTensor);
    auto dtype = result->GetDataType();
    auto Tshapedtype = shapeTensor->GetDataType();
    static internal::AicpuTaskSpace space("StatelessRandomNormalV2", ge::DEPEND_CONST_VALUE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        StatelessRandomNormalV2, OP_ATTR_NAMES({"dtype", "Tshape"}),
        OP_INPUT(shapeTensor, keyTensor, counterTensor, algTensor), OP_OUTPUT(outTensor), OP_ATTR(dtype, Tshapedtype));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return outTensor;
}

const aclTensor* StatelessRandomNormalV2(
    const aclTensor* result, const aclIntArray* key, const aclIntArray* counter, const aclTensor* alg,
    aclOpExecutor* executor)
{
    auto outTensor = executor->AllocTensor(result->GetViewShape(), result->GetDataType(), result->GetViewFormat());

    auto sizeV = op::ToShapeVector(result->GetViewShape());
    auto shapeTensor = executor->ConvertToTensor(sizeV.data(), sizeV.size(), op::ToOpDataType(ACL_INT64));

    auto keyTensor = executor->ConvertToTensor(key, op::ToOpDataType(ACL_UINT64));
    auto counterTensor = executor->ConvertToTensor(counter, op::ToOpDataType(ACL_UINT64));

    if (IsAiCoreSupport(outTensor->GetDataType())) {
        return StatelessRandomNormalV2AiCore(result, shapeTensor, keyTensor, counterTensor, alg, outTensor, executor);
    } else {
        return StatelessRandomNormalV2AiCpu(result, shapeTensor, keyTensor, counterTensor, alg, outTensor, executor);
    }
}

const aclTensor* StatelessRandomNormalV2(
    const aclTensor* result,
    const aclTensor* keyTensor, const aclTensor* counterTensor,
    const aclTensor* alg, aclOpExecutor* executor)
{
    auto outTensor = executor->AllocTensor(result->GetViewShape(), result->GetDataType(), result->GetViewFormat());

    auto sizeV = op::ToShapeVector(result->GetViewShape());
    auto shapeTensor = executor->ConvertToTensor(sizeV.data(), sizeV.size(), op::ToOpDataType(ACL_INT64));

    if (IsAiCoreSupport(outTensor->GetDataType())) {
        return StatelessRandomNormalV2AiCore(result, shapeTensor, keyTensor, counterTensor, alg, outTensor, executor);
    } else {
        return StatelessRandomNormalV2AiCpu(result, shapeTensor, keyTensor, counterTensor, alg, outTensor, executor);
    }
}
} // namespace l0op