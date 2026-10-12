/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stateless_random_normal_v3.h"
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

OP_TYPE_REGISTER(StatelessRandomNormalV3);

// AICORE算子kernel
static inline const aclTensor* StatelessRandomNormalV3AiCore(
    const aclTensor* result, const aclTensor* shapeTensor, const aclTensor* keyTensor, const aclTensor* counterTensor,
    const aclTensor* meanTensor, const aclTensor* stdevTensor,
    aclTensor* outTensor, aclOpExecutor* executor)
{
    L0_DFX(StatelessRandomNormalV3AiCore, shapeTensor, keyTensor, counterTensor, meanTensor, stdevTensor, outTensor);
    auto dtype = result->GetDataType();
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessRandomNormalV3, OP_INPUT(shapeTensor, keyTensor, counterTensor, meanTensor, stdevTensor),
        OP_OUTPUT(outTensor), OP_ATTR(dtype));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return outTensor;
}

const aclTensor* StatelessRandomNormalV3(
    const aclTensor* result, const aclTensor* key, const aclTensor* counter,
    const aclTensor* mean, const aclTensor* stdev, aclOpExecutor* executor)
{
    auto outTensor = executor->AllocTensor(result->GetViewShape(), result->GetDataType(), result->GetViewFormat());

    auto sizeV = op::ToShapeVector(result->GetViewShape());
    auto shapeTensor = executor->ConvertToTensor(sizeV.data(), sizeV.size(), op::ToOpDataType(ACL_INT64));

    return StatelessRandomNormalV3AiCore(result, shapeTensor, key, counter, mean, stdev, outTensor, executor);
}

const aclTensor* StatelessRandomNormalV3(
    const aclTensor* result, const aclIntArray* key, const aclIntArray* counter,
    const aclTensor* mean, const aclTensor* stdev, aclOpExecutor* executor)
{
    auto outTensor = executor->AllocTensor(result->GetViewShape(), result->GetDataType(), result->GetViewFormat());

    auto sizeV = op::ToShapeVector(result->GetViewShape());
    auto shapeTensor = executor->ConvertToTensor(sizeV.data(), sizeV.size(), op::ToOpDataType(ACL_INT64));

    auto keyTensor = executor->ConvertToTensor(key, op::ToOpDataType(ACL_UINT64));
    auto counterTensor = executor->ConvertToTensor(counter, op::ToOpDataType(ACL_UINT64));

    return StatelessRandomNormalV3AiCore(result, shapeTensor, keyTensor, counterTensor, mean, stdev, outTensor, executor);
}

} // namespace l0op