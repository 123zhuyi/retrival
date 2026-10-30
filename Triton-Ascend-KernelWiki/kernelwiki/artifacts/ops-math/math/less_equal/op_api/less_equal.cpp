/**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
OP_TYPE_REGISTER(LessEqual);

// AICORE
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_UINT64};

// 950支持类型
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT32,
    op::DataType::DT_INT64,   op::DataType::DT_INT8,   op::DataType::DT_UINT8,
    op::DataType::DT_BF16,    op::DataType::DT_UINT64, op::DataType::DT_BOOL};

// AICPU TF
static const std::initializer_list<op::DataType> AICPU_TF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT16, op::DataType::DT_UINT16, op::DataType::DT_DOUBLE};

// 根据dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        return CheckType(self->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
    }
    if (npuArch == NpuArch::DAV_3102) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* LessEqualAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* leOut, aclOpExecutor* executor)
{
    L0_DFX(LessEqualAiCore, self, other, leOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将LessEqual算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LessEqual, OP_INPUT(self, other), OP_OUTPUT(leOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LessEqualAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return leOut;
}

// AICPU算子 TF
static const aclTensor* LessEqualAiCpuTf(
    const aclTensor* self, const aclTensor* other, aclTensor* leOut, aclOpExecutor* executor)
{
    L0_DFX(LessEqualAiCpuTf, self, other);

    // 走TF
    static internal::AicpuTaskSpace space("LessEqual", ge::DEPEND_IN_SHAPE, true);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将LessEqual算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        LessEqual, OP_ATTR_NAMES({"T", "incompatible_shape_error"}), OP_INPUT(self, other), OP_OUTPUT(leOut),
        OP_ATTR(self->GetDataType(), true));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LessEqualAiCpuTf ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return leOut;
}

const aclTensor* LessEqual(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape outShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto leOut = executor->AllocTensor(outShape, op::DataType::DT_BOOL, op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
        return LessEqualAiCore(self, other, leOut, executor);
    } else {
        return LessEqualAiCpuTf(self, other, leOut, executor);
    }
}
} // namespace l0op
