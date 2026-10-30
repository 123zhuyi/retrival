/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ones_like.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(OnesLike);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_UINT8, op::DataType::DT_BF16,    op::DataType::DT_BOOL};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* OnesLikeAiCore(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(OnesLikeAiCore, self, out);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(OnesLike, OP_INPUT(self), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "OnesLike ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

// AICPU算子kernel
static const aclTensor* OnesLikeAiCPU(const aclTensor* self, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(OnesLikeAiCPU, self, out);
    if (IsComplexType(self->GetDataType()) || (self->GetDataType() == ge::DataType::DT_BF16)) {
        static internal::AicpuTaskSpace space("OnesLike", ge::DEPEND_IN_SHAPE, true);
        auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
            OnesLike, OP_ATTR_NAMES({"T"}), OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(self->GetDataType()));
        CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
        return out;
    }
    static internal::AicpuTaskSpace space("OnesLike");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(OnesLike, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* OnesLike(const aclTensor* self, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    CHECK_RET(out != nullptr, nullptr);
    if (out->IsEmpty()) {
        return out;
    }
    if (IsAiCoreSupport(self)) {
        return OnesLikeAiCore(self, out, executor);
    } else {
        CHECK_RET(IsAiCpuSupport(self), nullptr);
        return OnesLikeAiCPU(self, out, executor);
    }
}
} // namespace l0op
