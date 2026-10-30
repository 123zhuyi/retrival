/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "real.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Real);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_COMPLEX32, DataType::DT_COMPLEX64};

// 根据芯片类型、dtype判断算子是否支持aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* RealAiCore(
    const aclTensor* self, const aclTensor* out, DataType dstDtype, aclOpExecutor* executor)
{
    L0_DFX(RealAiCore, self, out, dstDtype);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Real算子加入任务队列
    // Real是算子的OpType，self是算子的输入，out是算子的输出
    auto retAicore =
        ADD_TO_LAUNCHER_LIST_AICORE(Real, OP_ATTR_NAMES({"Tout"}), OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(dstDtype));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "Real ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

// AICPU算子kernel
static const aclTensor* RealAiCpu(
    const aclTensor* self, const aclTensor* out, DataType dstDtype, aclOpExecutor* executor)
{
    L0_DFX(RealAiCpu, self, out, dstDtype);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Real算子加入任务队列
    // Real是算子的OpType，self是算子的输入，out是算子的输出
    static internal::AicpuTaskSpace space("Real", ge::DEPEND_IN_SHAPE, true);
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICPU(Real, OP_ATTR_NAMES({"Tout"}), OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(dstDtype));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* Real(const aclTensor* self, aclOpExecutor* executor)
{
    Shape outShape = self->GetViewShape();

    op::DataType dstDtype = self->GetDataType();
    ;
    if (self->GetDataType() == DataType::DT_COMPLEX64) {
        dstDtype = DataType::DT_FLOAT;
    } else if (self->GetDataType() == DataType::DT_COMPLEX32) {
        dstDtype = DataType::DT_FLOAT16;
    } else if (self->GetDataType() == DataType::DT_COMPLEX128) {
        dstDtype = DataType::DT_DOUBLE;
    }

    // 根据推导出的输出shape申请输出tensor
    // 第一个参数是输出shape，第二个参数是输出的dtype
    auto out = executor->AllocTensor(outShape, dstDtype);

    if (IsAiCoreSupport(self)) {
        return RealAiCore(self, out, dstDtype, executor);
    } else {
        return RealAiCpu(self, out, dstDtype, executor);
    }

    return out;
}

} // namespace l0op
