/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arange.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Range);
constexpr double eps = 0.0000000000001;
static const std::initializer_list<DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_BF16, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND310B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(DataType outType)
{
    // 获取芯片类型
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return CheckType(outType, ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310B) {
        return CheckType(outType, ASCEND310B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(outType, ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ArangeAiCore(
    const aclTensor* start, const aclTensor* end, const aclTensor* step, aclTensor* arangeOut, const bool isClosed,
    aclOpExecutor* executor)
{
    L0_DFX(ArangeAiCore, start, end, step, arangeOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Arange算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Range, OP_INPUT(start, end, step), OP_OUTPUT(arangeOut), OP_ATTR(isClosed));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArangeAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return arangeOut;
}

// AICPU算子kernel
static const aclTensor* ArangeAiCpu(
    const aclTensor* start, const aclTensor* end, const aclTensor* step, aclTensor* arangeOut, aclOpExecutor* executor)
{
    L0_DFX(ArangeAiCpu, start, end, step, arangeOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Arange算子加入任务队列
    static internal::AicpuTaskSpace space("Range", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Range, OP_ATTR_NAMES({"Tidx"}), OP_INPUT(start, end, step), OP_OUTPUT(arangeOut),
        OP_ATTR(start->GetDataType()));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArangeAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return arangeOut;
}

const aclTensor* Arange(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out, const bool isClosed,
    aclOpExecutor* executor)
{
    DataType outType = out->GetDataType();
    DataType inputType = outType;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910 ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P ||
        IsRegBase()) {
        if (outType == DataType::DT_INT64) {
            inputType = DataType::DT_INT64;
        } else {
            inputType = DataType::DT_DOUBLE;
        }
    }

    auto startTensor = executor->ConvertToTensor(start, inputType);
    auto endTensor = executor->ConvertToTensor(end, inputType);
    auto stepTensor = executor->ConvertToTensor(step, inputType);

    auto arangeOpOut = executor->AllocTensor(out->GetViewShape(), outType);

    if (IsAiCoreSupport(outType)) {
        return ArangeAiCore(startTensor, endTensor, stepTensor, arangeOpOut, isClosed, executor);
    } else {
        if (isClosed) {
            endTensor = executor->ConvertToTensor(executor->AllocScalar(end->ToDouble() + eps), inputType);
        }
        return ArangeAiCpu(startTensor, endTensor, stepTensor, arangeOpOut, executor);
    }

    return arangeOpOut;
}
} // namespace l0op
