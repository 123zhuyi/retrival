/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cross.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Cross);
OP_TYPE_REGISTER(CrossV2);

static const int64_t FORMER_LLENGTH_BOUND = 10240;
static const int64_t TAIL_LLENGTH_BOUND = 64;

static const std::initializer_list<op::DataType> REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {	 
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,    op::DataType::DT_INT8,	 
    op::DataType::DT_INT16,      op::DataType::DT_INT32,      op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> V2_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,      op::DataType::DT_INT32,
    op::DataType::DT_BF16};

// 根据芯片类型、dtype判断算子是否支持走aicore
inline static bool IsAiCoreSupport(const aclTensor *self)
{
    // Cross只需要判断dtype是否支持即可
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        return CheckType(self->GetDataType(), REGBASE_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsV2AiCoreSupport(const aclTensor *self) {
    if (IsRegBase()){
        return false;
    }
    // 获取芯片类型
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
        op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        return CheckType(self->GetDataType(), V2_AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

// AICORE算子kernel
inline static const aclTensor *CrossAiCore(const aclTensor *self, const aclTensor *other, int64_t dim,
    aclTensor *crossOut, aclOpExecutor *executor)
{
    L0_DFX(CrossAiCore, self, other, dim, crossOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Cross算子加入任务队列
    // Cross是算子的OpType，self, other是算子的输入，crossOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cross, OP_ATTR_NAMES({"dim"}), OP_INPUT(self, other), OP_OUTPUT(crossOut),
        OP_ATTR(dim));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CrossAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return crossOut;
}

// AICORE算子kernel
inline static const aclTensor *CrossV2AiCore(const aclTensor *self, const aclTensor *other, int64_t dim,
    aclTensor *crossOut, aclOpExecutor *executor)
{
    L0_DFX(CrossV2AiCore, self, other, crossOut, dim);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore CrossV2算子加入任务队列
    // CrossV2是算子的OpType，self, other是算子的输入，crossOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CrossV2, OP_INPUT(self, other), OP_OUTPUT(crossOut),
        OP_ATTR(dim));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CrossV2AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return crossOut;
}

// AICPU算子kernel
inline static const aclTensor *CrossAiCpu(const aclTensor *self, const aclTensor *other, int64_t dim,
    aclTensor *crossOut, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_CPU_LAUNCHER_LIST，将AiCpu Cross算子加入任务队列
    // Cross是算子的OpType，self,other是算子的输入，crossOut是算子的输出
    L0_DFX(CrossAiCpu, self, other, dim, crossOut);
    static internal::AicpuTaskSpace space("Cross");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Cross, OP_ATTR_NAMES({"dim"}), OP_INPUT(self, other), OP_OUTPUT(crossOut),
                                          OP_ATTR(dim));
    OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CrossAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return crossOut;
}

const aclTensor *Cross(const aclTensor *self, const aclTensor *other, int64_t dim, aclOpExecutor *executor)
{
    Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    auto crossOut = executor->AllocTensor(broadcastShape, self->GetDataType(), op::Format::FORMAT_ND);
    if (IsV2AiCoreSupport(self)) {
        return CrossV2AiCore(self, other, dim, crossOut, executor);
    } else if (IsAiCoreSupport(self)) {
        return CrossAiCore(self, other, dim, crossOut, executor);
    } else {
        return CrossAiCpu(self, other, dim, crossOut, executor);
    }
}
}
