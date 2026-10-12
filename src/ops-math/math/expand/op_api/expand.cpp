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
 * \file expand.cpp
 * \brief
 */
#include "expand.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Expand);

static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_UINT8,
    DataType::DT_INT8,  DataType::DT_BOOL,    DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_UINT8,
    DataType::DT_INT8,  DataType::DT_BOOL,    DataType::DT_BF16,  DataType::DT_INT64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if ((curArch == NpuArch::DAV_2201) || (curArch == NpuArch::DAV_3510)) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ExpandAiCore(
    const aclTensor* self, const aclTensor* shape, const aclTensor* expandOut, aclOpExecutor* executor)
{
    L0_DFX(ExpandAiCore, self, shape, expandOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Expand算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Expand, OP_INPUT(self, shape), OP_OUTPUT(expandOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ExpandAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return expandOut;
}

// AICPU算子kernel
static const aclTensor* ExpandAiCpu(
    const aclTensor* self, const aclTensor* shape, const aclTensor* expandOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Expand算子加入任务队列
    L0_DFX(ExpandAiCpu, self, shape);

    static internal::AicpuTaskSpace space("Expand");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Expand, OP_ATTR_NAMES(), OP_INPUT(self, shape), OP_OUTPUT(expandOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ExpandAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return expandOut;
}

bool ExpandInfershape(const aclTensor* x, const op::Shape& shape, op::Shape& expandShape)
{
    if (x->GetViewShape() == shape && x->GetStorageShape() == shape && x->GetOriginalShape() == shape) {
        expandShape = shape;
        return true;
    }

    auto expandDimNum = static_cast<int64_t>(shape.GetDimNum());
    auto xShape = x->GetViewShape();
    auto xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    expandShape.SetDimNum(expandDimNum);
    int64_t lenDiff = expandDimNum - xDimNum;

    if (lenDiff < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "length of shape should not be less than input_x's dimension.");
        return false;
    }

    for (int64_t i = 0; i < xDimNum; i++) {
        if (shape[lenDiff + i] == -1) {
            expandShape[lenDiff + i] = xShape[i];
        }
        if ((xShape[i] != shape[lenDiff + i]) && (xShape[i] != 1)) {
            OP_LOGE(ACL_ERROR_INVALID_PARAM, "shape cannot be expanded to an incompatible value.");
            return false;
        }
        expandShape[lenDiff + i] = shape[lenDiff + i];
    }
    for (int64_t i = 0; i < lenDiff; i++) {
        expandShape[i] = shape[i];
    }
    return true;
}

const aclTensor* Expand(const aclTensor* self, const aclIntArray* shape, aclOpExecutor* executor)
{
    L0_DFX(Expand, self, shape);
    if (!op::IsContiguous(self)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input tensor should be contiguous.");
        return nullptr;
    }
    op::Shape newShape;
    for (int64_t i = 0; i < static_cast<int64_t>(shape->Size()); i++) {
        newShape.AppendDim((*shape)[i]);
    }
    op::Shape expandShape;
    if (!ExpandInfershape(self, newShape, expandShape)) {
        OP_LOGE(ACL_ERROR_FAILURE, "InferShape failed for expand operation.");
        return nullptr;
    }

    auto expandOut = executor->AllocTensor(expandShape, self->GetDataType());
    CHECK_RET(expandOut != nullptr, nullptr);
    auto shapeTensor = executor->ConvertToTensor(shape, DataType::DT_INT64);
    CHECK_RET(shapeTensor != nullptr, nullptr);

    // 根据推导出的输出shape申请输出tensor
    if (IsAiCoreSupport(self)) {
        return ExpandAiCore(self, shapeTensor, expandOut, executor);
    } else {
        return ExpandAiCpu(self, shapeTensor, expandOut, executor);
    }
}
} // namespace l0op
