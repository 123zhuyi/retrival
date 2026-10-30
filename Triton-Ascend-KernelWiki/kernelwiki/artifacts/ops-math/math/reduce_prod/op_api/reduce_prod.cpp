/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_prod.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ReduceProd);

static const std::initializer_list<op::DataType> AICORE_950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT64, op::DataType::DT_BF16,    op::DataType::DT_BOOL, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT64, op::DataType::DT_BF16,    op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64,   op::DataType::DT_BF16};

// 根据芯片类型和数据类型判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // ReduceProd算子只需要判断数据类型
    if (IsRegBase()) {
        return CheckType(self->GetDataType(), AICORE_950_DTYPE_SUPPORT_LIST);
    } else if (
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), AICORE_910B_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICPU算子kernel
static const aclTensor* ReduceProdAICpu(
    const aclTensor* self, const aclTensor* axes, const bool keepDims, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ReduceProdAICpu, self, axes, keepDims, out);
    static internal::AicpuTaskSpace space("Prod", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        ReduceProd, OP_ATTR_NAMES({"keep_dims", "Tidx"}), OP_INPUT(self, axes), OP_OUTPUT(out),
        OP_ATTR(keepDims, axes->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

// AICORE算子kernel
static const aclTensor* ReduceProdAICore(
    const aclTensor* self, const aclTensor* axes, const bool keepDims, bool noopWithEmptyAxes, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(ReduceProdAICore, self, axes, keepDims, noopWithEmptyAxes, out);
    ADD_TO_LAUNCHER_LIST_AICORE(ReduceProd, OP_INPUT(self, axes), OP_OUTPUT(out), OP_ATTR(keepDims, noopWithEmptyAxes));
    return out;
}

const aclTensor* ReduceProd(const aclTensor* self, const aclTensor* axes, const bool keepDims, aclOpExecutor* executor)
{
    if (self->GetViewShape().GetDimNum() == 0) {
        return self;
    }
    auto out = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);

    // dim为空时，默认保留所有轴
    bool noopWithEmptyAxes = true;
    INFER_SHAPE(ReduceProd, OP_INPUT(self, axes), OP_OUTPUT(out), OP_ATTR(keepDims, noopWithEmptyAxes));

    if (IsAiCoreSupport(self)) {
        return ReduceProdAICore(self, axes, keepDims, noopWithEmptyAxes, out, executor);
    }
    return ReduceProdAICpu(self, axes, keepDims, out, executor);
}
} // namespace l0op