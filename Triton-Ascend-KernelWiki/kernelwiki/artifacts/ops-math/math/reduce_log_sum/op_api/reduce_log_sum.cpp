/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_log_sum.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceLogSum);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16, };
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

//根据芯片类型，dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(DataType inputDtype)
{
    if (IsRegBase()) {
        return CheckType(inputDtype, DTYPE_SUPPORT_LIST_950);
    }
    //只需要判断dtype
    return CheckType(inputDtype, AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static inline const aclTensor* ReduceLogSumAiCore(
    const aclTensor* data, const aclTensor* axes, aclTensor* reduce, bool keepDims, aclOpExecutor* executor)
{
    L0_DFX(ReduceLogSumAiCore, data, axes, reduce, keepDims);
    //使用框架宏ADD_TO_LAUNCHER_LIST_AICORE,将AiCore ReduceLogSum算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ReduceLogSum, OP_INPUT(data, axes), OP_OUTPUT(reduce), OP_ATTR(keepDims));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceLogSumAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);

    return reduce;
}

const aclTensor* ReduceLogSum(const aclTensor* data, const aclIntArray* axes, bool keepDims, aclOpExecutor* executor)
{
    auto dims = executor->ConvertToTensor(axes, op::DataType::DT_INT64);
    auto output = executor->AllocTensor(data->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    INFER_SHAPE(ReduceLogSum, OP_INPUT(data, dims), OP_OUTPUT(output), OP_ATTR(keepDims));

    op::Shape outShape = data->GetViewShape();
    auto count = axes->Size();
    size_t dimNum = outShape.GetDimNum();
    if (keepDims) {
        for (uint64_t i = 0; i < count; i++) {
            int64_t dimIndex = static_cast<int64_t>((*axes)[i]);
            int64_t dimNew = dimIndex >= 0 ? dimIndex : dimIndex + dimNum;
            outShape.SetDim(dimNew, 1);
        }
        output->SetViewShape(outShape);
    }

    if (IsAiCoreSupport(data->GetDataType())) {
        return ReduceLogSumAiCore(data, dims, output, keepDims, executor);
    }
    return output;
}
} // namespace l0op
