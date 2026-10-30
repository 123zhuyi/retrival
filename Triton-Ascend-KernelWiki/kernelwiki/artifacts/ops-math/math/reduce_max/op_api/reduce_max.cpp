/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_max.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceMax);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_UINT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_GE910B = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_UINT8, op::DataType::DT_INT32,
    op::DataType::DT_INT64,   op::DataType::DT_BF16,  op::DataType::DT_INT8};

// 判断芯片类型是否大于等于910B
static inline bool CheckSocVersionGe910B(void)
{
    return (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
            GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) ||
           IsRegBase();
}

static bool IsAiCoreSupport(const aclTensor* self)
{
    // 获取芯片类型
    if (CheckSocVersionGe910B()) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_GE910B);
    }
    // 910
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_910);
}

// AICORE算子kernel
static const aclTensor* ReduceMaxAiCore(
    const aclTensor* self, const aclTensor* dimList, bool keepDim, bool noopWithEmptyDims, const aclTensor* maxOut,
    aclOpExecutor* executor)
{
    L0_DFX(ReduceMaxAiCore, self, dimList, keepDim, noopWithEmptyDims, maxOut);

    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        ReduceMax, OP_INPUT(self, dimList), OP_OUTPUT(maxOut), OP_ATTR(keepDim, noopWithEmptyDims));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ReduceMax ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return maxOut;
}

// AICPU算子kernel
static const aclTensor* ReduceMaxAiCpu(
    const aclTensor* self, const aclTensor* dimList, bool keepDim, const aclTensor* maxOut, aclOpExecutor* executor)
{
    L0_DFX(ReduceMaxAiCpu, self, dimList, keepDim, maxOut);

    static internal::AicpuTaskSpace space("Max", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        ReduceMax, OP_ATTR_NAMES({"keep_dims", "Tidx"}), OP_INPUT(self, dimList), OP_OUTPUT(maxOut),
        OP_ATTR(keepDim, dimList->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return maxOut;
}

const aclTensor* ReduceMax(
    const aclTensor* self, const aclIntArray* dim, bool keepDim, bool noopWithEmptyDims, aclOpExecutor* executor)
{
    auto dimList = executor->ConvertToTensor(dim, op::DataType::DT_INT64);
    auto maxOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());

    auto ret = INFER_SHAPE(ReduceMax, OP_INPUT(self, dimList), OP_OUTPUT(maxOut), OP_ATTR(keepDim, noopWithEmptyDims));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ReduceMax infer shape faild.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
        return ReduceMaxAiCore(self, dimList, keepDim, noopWithEmptyDims, maxOut, executor);
    } else {
        return ReduceMaxAiCpu(self, dimList, keepDim, maxOut, executor);
    }
}
} // namespace l0op
