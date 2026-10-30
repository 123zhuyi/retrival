/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "less.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Less);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_INT32,   op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE};

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
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* LessAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* ltOut, aclOpExecutor* executor)
{
    L0_DFX(LessAiCore, self, other, ltOut);
    // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将Less算子加入任务队列
    // op::OP_TYPE_Less是算子的OpType，self、other是算子的输入，lt_out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Less, OP_INPUT(self, other), OP_OUTPUT(ltOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LessAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return ltOut;
}

// AICPU算子kernel
static const aclTensor* LessAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* ltOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将Less算子加入任务队列
    L0_DFX(LessAiCpu, self, other, ltOut);

    static internal::AicpuTaskSpace space("Less");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Less, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(ltOut));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LessAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return ltOut;
}

const aclTensor* Less(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    // 根据Less算子语义，通过输入shape推导算子输出shape
    // 不同算子语义不同，推导的逻辑也不同，需要根据算子具体功能实现推导逻辑
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    // 根据推导出的输出shape申请输出tensor
    // 第一个参数是输出shape，第二个参数是输出的dtype
    auto ltOut = executor->AllocTensor(broadcastShape, op::DataType::DT_BOOL);

    if (IsAiCoreSupport(self) && IsAiCoreSupport(other)) {
        return LessAiCore(self, other, ltOut, executor);
    } else {
        return LessAiCpu(self, other, ltOut, executor);
    }
}
} // namespace l0op
