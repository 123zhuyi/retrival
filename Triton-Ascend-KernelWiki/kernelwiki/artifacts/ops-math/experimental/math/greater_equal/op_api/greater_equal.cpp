/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "greater_equal.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GreaterEqual);

// 仅1971支持DT_BF16
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BF16,  op::DataType::DT_INT64};

// 610lite支持类型
static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, op::DataType::DT_INT8, op::DataType::DT_UINT8};

// 950支持类型
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_BF16,
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        return CheckType(self->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
    }
    if (npuArch == NpuArch::DAV_3102) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    return op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

bool IsGreaterEqualSupportNonContiguous(const aclTensor* self)
{
    bool isSupportNonContiguous = IsRegBase();
    return isSupportNonContiguous && IsAiCoreSupport(self);
}

static const aclTensor* GreaterEqualAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(GreaterEqualAiCore, self, other, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GreaterEqual, OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterEqualAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

static const aclTensor* GreaterEqualAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(GreaterEqualAiCpu, self, other, out);
    static internal::AicpuTaskSpace space("GreaterEqual");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(GreaterEqual, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GreaterEqualAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* GreaterEqual(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(GreaterEqual, self, other);

    op::Shape outShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "InferShape %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto out = executor->AllocTensor(outShape, op::DataType::DT_BOOL, op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
        return GreaterEqualAiCore(self, other, out, executor);
    }
    return GreaterEqualAiCpu(self, other, out, executor);
}
} // namespace l0op
