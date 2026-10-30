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
 * \file roll.cpp
 * \brief
 */

#include "roll.h"
#include "opdev/platform.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Roll);

static bool IsAiCoreSupport()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2002 || curArch == NpuArch::DAV_1001 ||
        curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
        return true;
    }

    return false;
}

aclTensor* RollAiCore(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclTensor* rollOut, aclOpExecutor* executor)
{
    L0_DFX(RollAiCore, x, shifts, dims);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Roll, OP_INPUT(x), OP_OUTPUT(rollOut), OP_ATTR(shifts, dims));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RollAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return rollOut;
}

aclTensor* RollAiCpu(
    const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclTensor* rollOut, aclOpExecutor* executor)
{
    L0_DFX(RollAiCpu, x, shifts, dims);

    static internal::AicpuTaskSpace space("Roll", ge::DEPEND_IN_SHAPE, true);
    op::DataType inputDtype = x->GetDataType();
    op::DataType int64Dtype = op::DataType::DT_INT64;
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Roll, OP_ATTR_NAMES({"T", "Tshift", "Taxis"}), OP_INPUT(x, shifts, dims), OP_OUTPUT(rollOut),
        OP_ATTR(inputDtype, int64Dtype, int64Dtype));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return rollOut;
}

const aclTensor* Roll(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclOpExecutor* executor)
{
    if (x == nullptr) {
        return nullptr;
    }
    auto rollOut = executor->AllocTensor(x->GetViewShape(), x->GetDataType(), x->GetViewFormat());
    if (IsAiCoreSupport())
        RollAiCore(x, shifts, dims, rollOut, executor);
    else
        return RollAiCpu(x, shifts, dims, rollOut, executor);
    return rollOut;
}

} // namespace l0op