/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bincount.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Bincount);

// AICORE 950支持类型
static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* weights)
{
    if (IsRegBase()) {
        return CheckType(weights->GetDataType(), ASCEND950_DTYPE_SUPPORT_LIST);
    }

    return false;
}

// AICPU算子kernel
static const aclTensor* BincountAiCpu(
    const aclTensor* x, const aclTensor* size, const aclTensor* weights, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(BincountAiCpu, x, size, weights, out);

    static internal::AicpuTaskSpace space("Bincount", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Bincount, OP_ATTR_NAMES({"T", "T"}), OP_INPUT(x, size, weights), OP_OUTPUT(out),
        OP_ATTR(weights->GetDataType(), out->GetDataType()));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BincountAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

// AICORE算子kernel
static const aclTensor* BincountAiCore(
    const aclTensor* x, const aclTensor* size, const aclTensor* weights, aclTensor* bins, aclOpExecutor* executor)
{
    L0_DFX(BincountAiCore, x, size, weights, bins);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Bincount, OP_INPUT(x, size, weights), OP_OUTPUT(bins));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BincountAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return bins;
}

const aclTensor* Bincount(const aclTensor* x, const aclTensor* weights, int64_t size, aclOpExecutor* executor)
{
    auto sizeTensor = executor->ConvertToTensor(executor->AllocScalar(size), op::DataType::DT_INT32);
    Shape outShape;
    outShape.SetDimNum(1);
    outShape.SetDim(0, size);

    auto out =
        executor->AllocTensor(outShape, outShape, weights->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    if (IsAiCoreSupport(weights)) {
        return BincountAiCore(x, sizeTensor, weights, out, executor);
    } else {
        return BincountAiCpu(x, sizeTensor, weights, out, executor);
    }
}
} // namespace l0op