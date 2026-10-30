/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "abs.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Abs);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND610LITE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32, DataType::DT_INT64};

static const std::initializer_list<op::DataType> ARCH_REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8,
    DataType::DT_INT16, DataType::DT_INT32,   DataType::DT_INT64,   DataType::DT_COMPLEX32,   DataType::DT_COMPLEX64};

// 根据芯片类型、dtype判断算子是否支持aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        return CheckType(self->GetDataType(), ARCH_REGBASE_AICORE_DTYPE_SUPPORT_LIST);
    }
    if (npuArch == NpuArch::DAV_2201) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    if (npuArch == NpuArch::DAV_3102) {
        return CheckType(self->GetDataType(), ASCEND610LITE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool AbsInferShape(const op::Shape& selfShape, op::Shape& outShape)
{
    outShape = selfShape;
    return true;
}

// AICORE算子kernel
static const aclTensor* AbsAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AbsAiCore, self, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Abs算子加入任务队列
    // Abs是算子的OpType，self是算子的输入，out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Abs, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AbsAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static const aclTensor* AbsAiCpu(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(AbsAiCpu, self, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Abs算子加入任务队列
    // Abs是算子的OpType，self是算子的输入，out是算子的输出
    static internal::AicpuTaskSpace space("Abs", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Abs, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AbsAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}

const aclTensor* Abs(const aclTensor* self, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;
    if (!AbsInferShape(self->GetViewShape(), outShape)) {
            OP_LOGE(ACL_ERROR_INVALID_PARAM, "infer shape failed.");
            return nullptr;
        }
    if(self->GetDataType() != ge::DT_COMPLEX64 && self->GetDataType() != ge::DT_COMPLEX32) {
        out = executor->AllocTensor(outShape, self->GetDataType());
    } else { 
        if (self->GetDataType() == ge::DT_COMPLEX32) {
            out = executor->AllocTensor(outShape, ge::DT_FLOAT16);
        } else if (self->GetDataType() == ge::DT_COMPLEX64) {
            out = executor->AllocTensor(outShape, ge::DT_FLOAT);
        }
    } 

    if (IsAiCoreSupport(self)) {
        return AbsAiCore(self, out, executor);
    } else {
        return AbsAiCpu(self, out, executor);
    }

    return out;
}

} // namespace l0op
