/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "equal.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Equal);

// 1980
static const std::initializer_list<op::DataType> AICORE910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,   op::DataType::DT_UINT8,
    op::DataType::DT_UINT64,  op::DataType::DT_UINT32, op::DataType::DT_BOOL};

// 610lite
static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

// 1971
static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

// 950
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_BOOL,  op::DataType::DT_BF16,
    op::DataType::DT_UINT64, op::DataType::DT_DOUBLE};

// AICPU TF
static const std::initializer_list<op::DataType> AICPU_TF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        return CheckType(self->GetDataType(), REGBASE_DTYPE_SUPPORT_LIST);
    }
    if (npuArch == NpuArch::DAV_2201) {
        return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
    }

    if (npuArch == NpuArch::DAV_3102) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    // 1980 & other
    return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* EqualAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* eqOut, aclOpExecutor* executor)
{
    L0_DFX(EqualAiCore, self, other, eqOut);
    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE，将Equal算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Equal, OP_INPUT(self, other), OP_OUTPUT(eqOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EqualAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return eqOut;
}

// AICPU算子kernel
static const aclTensor* EqualAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* eqOut, aclOpExecutor* executor)
{
    L0_DFX(EqualAiCpu, self, other);

    static internal::AicpuTaskSpace space("Equal");
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将Equal算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Equal, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(eqOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EqualAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return eqOut;
}

// AICPU算子 TF
static const aclTensor* EqualAiCpuTf(
    const aclTensor* self, const aclTensor* other, aclTensor* eqOut, aclOpExecutor* executor)
{
    L0_DFX(EqualAiCpuTf, self, other);

    // 走TF
    static internal::AicpuTaskSpace space("Equal", ge::DEPEND_IN_SHAPE, true);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将Equal算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Equal, OP_ATTR_NAMES({"T", "incompatible_shape_error"}), OP_INPUT(self, other), OP_OUTPUT(eqOut),
        OP_ATTR(self->GetDataType(), true));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EqualAiCpuTf ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return eqOut;
}

const aclTensor* Equal(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape outShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), outShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    auto eqOut = executor->AllocTensor(outShape, op::DataType::DT_BOOL, op::Format::FORMAT_ND);

    if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
        return EqualAiCore(self, other, eqOut, executor);
    } else {
        if (CheckType(self->GetDataType(), AICPU_TF_DTYPE_SUPPORT_LIST) ||
            CheckType(other->GetDataType(), AICPU_TF_DTYPE_SUPPORT_LIST)) {
            return EqualAiCpuTf(self, other, eqOut, executor);
        }
        return EqualAiCpu(self, other, eqOut, executor);
    }
}
} // namespace l0op
