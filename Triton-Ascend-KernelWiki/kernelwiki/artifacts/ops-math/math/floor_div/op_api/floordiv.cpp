/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "floordiv.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FloorDiv);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_BF16,  op::DataType::DT_INT64};

static inline const std::initializer_list<op::DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_3510:
        case NpuArch::DAV_2201: {
            return REGBASE_DTYPE_SUPPORT_LIST;
        }
        default: {
            return AICORE_DTYPE_SUPPORT_LIST;
        }
    }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    return op::CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

// AICORE算子kernel
static const aclTensor* FloorDivAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* floorDivOut, aclOpExecutor* executor)
{
    L0_DFX(FloorDivAiCore);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FloorDiv, OP_INPUT(self, other), OP_OUTPUT(floorDivOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FloorDivAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return floorDivOut;
}

// AICPU-tf算子kernel
static const aclTensor* FloorDivAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* floorDivOut, aclOpExecutor* executor)
{
    L0_DFX(FloorDivAiCpu);
    static internal::AicpuTaskSpace space("FloorDiv", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(FloorDiv, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(floorDivOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FloorDivAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return floorDivOut;
}

const aclTensor* FloorDiv(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (IsAiCoreSupport(self)) {
        return FloorDivAiCore(self, other, out, executor);
    } else {
        return FloorDivAiCpu(self, other, out, executor);
    }
}

const aclTensor* FloorDiv(const aclTensor* self, const aclTensor* other, bool isScalar, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    
    aclTensor* out;
    if (isScalar || self->GetDataType() == other->GetDataType()) {
        out = executor->AllocTensor(broadcastShape, self->GetDataType());
    } else {
        out = executor->AllocTensor(broadcastShape, op::DataType::DT_FLOAT);
    }

    if (IsAiCoreSupport(self)) {
        return FloorDivAiCore(self, other, out, executor);
    } else {
        return FloorDivAiCpu(self, other, out, executor);
    }
}

} // namespace l0op
