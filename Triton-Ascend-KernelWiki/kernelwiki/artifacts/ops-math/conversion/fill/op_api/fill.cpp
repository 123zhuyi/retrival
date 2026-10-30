/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "fill.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"
#include <map>

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Fill);
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_INT8,    op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_GE910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_INT8,  op::DataType::DT_BOOL,    op::DataType::DT_BF16, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64};

// 判断芯片类型是否大于等于910B
static inline bool CheckSocVersionGe910B(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

// 根据芯片类型、dtype判断算子是否支持走aicore
inline static bool IsAiCoreSupport(const aclTensor* self)
{
    // 获取芯片类型
    if (CheckSocVersionGe910B()) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_GE910B);
    }
    // 910
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910);
}

// AICORE算子kernel
inline static const aclTensor* FillAiCore(
    const aclTensor* dims, const aclTensor* value, aclTensor* fillOut, aclOpExecutor* executor)
{
    L0_DFX(FillAiCore, dims, value, fillOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Fill算子加入任务队列
    // Fill是算子的OpType，dims、value是算子的输入，fillOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Fill, OP_INPUT(dims, value), OP_OUTPUT(fillOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FillAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return fillOut;
}

// AICPU算子kernel
static const aclTensor* FillAiCpu(
    const aclTensor* dims, const aclTensor* value, aclTensor* fillOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Fill算子加入任务队列
    // Fill是算子的OpType，dims、value是算子的输入，fillOut是算子的输出
    L0_DFX(FillAiCpu, dims, value);

    static internal::AicpuTaskSpace space("Fill", ge::DEPEND_CONST_VALUE, true);
    op::DataType index_type = dims->GetDataType();
    op::DataType value_type = value->GetDataType();
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Fill, OP_ATTR_NAMES({"index_type", "T"}), OP_INPUT(dims, value), OP_OUTPUT(fillOut),
        OP_ATTR(index_type, value_type));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FillAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return fillOut;
}

const aclTensor* Fill(
    const aclTensor* dims, const aclTensor* value, const aclIntArray* outShape, aclOpExecutor* executor)
{
    OP_LOGI("Entering l0 Fill");
    gert::Shape broadcast_shape;
    for (uint64_t i = 0; i < outShape->Size(); i++) {
        broadcast_shape.AppendDim((*outShape)[i]);
    }
    auto fillOut = executor->AllocTensor(broadcast_shape, value->GetDataType());
    if (fillOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Fill out tensor alloc failed");
        return nullptr;
    }
    if (fillOut->IsEmpty()) {
        return fillOut;
    }
    if (IsAiCoreSupport(value)) {
        return FillAiCore(dims, value, fillOut, executor);
    } else {
        return FillAiCpu(dims, value, fillOut, executor);
    }
}
} // namespace l0op
