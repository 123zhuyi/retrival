/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trace.cpp
 * \brief
 */

#include "trace.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Trace);

static bool IsAiCoreSupport(const aclTensor* self)
{
    // 获取芯片类型,判断是1971还是1980
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910_93) {
        return CheckType(
            self->GetDataType(), {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16});
    }

    // 1980 & other
    return CheckType(self->GetDataType(), {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16});
}

const aclTensor* TraceAiCpu(const aclTensor* self, aclOpExecutor* executor, aclTensor* out)
{
    L0_DFX(TraceAiCpu, self);
    // add Trace to LAUNCHER_LIST_AICPU
    static internal::AicpuTaskSpace space("Trace");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Trace, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* TraceAiCore(const aclTensor* self, aclOpExecutor* executor, aclTensor* out)
{
    L0_DFX(TraceAiCore, self);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Trace, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TraceAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

const aclTensor* Trace(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Trace, self);

    // 根据Trace算子语义，通过输入shape推导算子输出shape
    op::Shape outShape;
    outShape.SetDimNum(1);
    outShape.SetDim(0, 1);
    // 根据推导出的输出shape申请输出tensor
    auto outDtype = (IsIntegralType(self->GetDataType(), true)) ? op::DataType::DT_INT64 : self->GetDataType();
    auto out = executor->AllocTensor(outShape, outDtype);
    CHECK_RET(out != nullptr, nullptr);
    // Trace是算子的OpType，self是算子的输入，out是算子的输出
    if (IsAiCoreSupport(self)) {
        return TraceAiCore(self, executor, out);
    } else {
        return TraceAiCpu(self, executor, out);
    }
}
} // namespace l0op