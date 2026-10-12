/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pack.h"

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
OP_TYPE_REGISTER(Pack);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,      op::DataType::DT_INT16,  op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,     op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,  op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensorList* inputs)
{
    // Pack只需要判断dtype
    for (uint64_t i = 0; i < inputs->Size(); i++) {
        if (!CheckType((*inputs)[i]->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
            return false;
        }
    }
    return true;
}

// AICORE算子kernel
static const aclTensor* PackAiCore(
    const aclTensorList* inputs, int64_t dim, aclTensor* out, op::DataType out_dtype, aclOpExecutor* executor)
{
    L0_DFX(PackAiCore, inputs, dim, out, out_dtype);
    ADD_TO_LAUNCHER_LIST_AICORE(Pack, OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(dim));
    return out;
}

// AICPU算子kernel
static const aclTensor* PackAiCpu(
    const aclTensorList* inputs, int64_t dim, aclTensor* out, op::DataType out_dtype, aclOpExecutor* executor)
{
    L0_DFX(PackAiCpu, inputs, dim, out, out_dtype);
    static internal::AicpuTaskSpace space("Pack", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Pack, OP_ATTR_NAMES({"N", "axis"}), OP_INPUT(inputs), OP_OUTPUT(out), OP_ATTR(static_cast<int64_t>(inputs->Size()), dim));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

static bool PackInferShape(const aclTensorList* inputs, int64_t dim, op::Shape& pack_shape)
{
    auto shape = (*inputs)[0]->GetViewShape();
    size_t dim_num = shape.GetDimNum();
    pack_shape.SetDimNum(dim_num + 1);
    size_t inputs_size = inputs->Size();
    size_t dim_new = dim >= 0 ? dim : dim + dim_num + 1;
    for (size_t i = 0, j = 0; i <= dim_num; i++, j++) {
        if (i == dim_new) {
            pack_shape.SetDim(i, inputs_size);
            j--;
        } else {
            pack_shape.SetDim(i, shape.GetDim(j));
        }
    }
    return true;
}

const aclTensor* Pack(const aclTensorList* inputs, int64_t dim, op::DataType out_dtype, aclOpExecutor* executor)
{
    // 推导算子输出shape
    op::Shape pack_shape;
    if (!PackInferShape(inputs, dim, pack_shape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inference shape for pack failed.");
        return nullptr;
    }
    // 根据推导出的输出shape申请输出tensor
    auto out = executor->AllocTensor(pack_shape, out_dtype);

    if (IsAiCoreSupport(inputs)) {
        return PackAiCore(inputs, dim, out, out_dtype, executor);
    } else {
        return PackAiCpu(inputs, dim, out, out_dtype, executor);
    }
}
} // namespace l0op
