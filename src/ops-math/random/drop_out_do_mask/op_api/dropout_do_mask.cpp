/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dropout_do_mask.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(DropOutDoMask);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
    }
}

// 判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* input)
{
    return CheckType(input->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

// AICORE算子kernel
static const aclTensor* DropoutDoMaskAiCore(
    const aclTensor* input, const aclTensor* mask, const aclTensor* prob, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(DropoutDoMaskAiCore, input, mask, prob, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DropOutDoMask, OP_INPUT(input, mask, prob), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DropoutDoMaskAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AICPU算子kernel
static const aclTensor* DropoutDoMaskAiCpu(
    const aclTensor* input, const aclTensor* mask, const aclTensor* prob, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(DropoutDoMaskAiCpu, input, mask, prob, out);

    static internal::AicpuTaskSpace space("DropOutDoMask");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(DropOutDoMask, OP_ATTR_NAMES(), OP_INPUT(input, mask, prob), OP_OUTPUT(out));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DropoutDoMaskAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return out;
}
const aclTensor* DropoutDoMask(
    const aclTensor* input, const aclTensor* mask, const aclTensor* prob, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(input->GetViewShape(), input->GetDataType());
    if (IsAiCoreSupport(input)) {
        return DropoutDoMaskAiCore(input, mask, prob, out, executor);
    } else {
        return DropoutDoMaskAiCpu(input, mask, prob, out, executor);
    }
    return out;
}
} // namespace l0op