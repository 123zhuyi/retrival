/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log_space.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(LogSpace);

// Ascend950（DAV_3510）原有支持的输出 dtype
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_DAV_3510 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
// Atlas A2/A3（DAV_2201）新增支持：在原有浮点基础上增加整型输出
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_DAV_2201 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8,
    DataType::DT_INT16, DataType::DT_INT32,   DataType::DT_UINT8};

static bool IsAiCoreSupport(const aclTensor* result)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) { // Ascend950：原有实现
        OP_CHECK(CheckType(result->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_DAV_3510),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LogSpace unsupported dtype: %d (need FLOAT/FLOAT16/BFLOAT16)",
                         static_cast<int>(result->GetDataType())),
                 return false);
        return true;
    }
    if (npuArch == NpuArch::DAV_2201) { // Atlas A2/A3：本次新增支持
        OP_CHECK(CheckType(result->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_DAV_2201),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "LogSpace unsupported dtype: %d "
                         "(need FLOAT/FLOAT16/BFLOAT16/INT8/INT16/INT32/UINT8)",
                         static_cast<int>(result->GetDataType())),
                 return false);
        return true;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "LogSpace only supports DAV_3510 (Ascend950) / DAV_2201 (Atlas A2/A3), got npuArch=%d",
            static_cast<int>(npuArch));
    return false;
}

static const aclTensor* LogSpaceAiCore(float startF, float endF, int64_t steps, float baseF, const aclTensor* out,
                                       aclOpExecutor* executor)
{
    L0_DFX(LogSpaceAiCore, startF, endF, steps, baseF, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LogSpace, OP_INPUT(), OP_OUTPUT(out), OP_ATTR(startF, endF, steps, baseF));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "LogSpaceAiCore failed."), return nullptr);
    return out;
}

const aclTensor* LogSpace(float startF, float endF, int64_t steps, float baseF, const aclTensor* result,
                          aclOpExecutor* executor)
{
    OP_CHECK(IsAiCoreSupport(result), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    // 输出 tensor 由 L2 提供（已是用户传入的 result）。L0 直接复用。
    return LogSpaceAiCore(startF, endF, steps, baseF, result, executor);
}

} // namespace l0op
