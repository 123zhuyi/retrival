/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_acosh.cpp
 * @brief ACLNN L2 API 实现 - Acosh 算子
 *
 * 标准两段式流程：
 * 1. aclnnAcoshGetWorkspaceSize: CREATE_EXECUTOR -> CheckParams -> HasEmptyTensor
 *    -> Contiguous -> l0op::Acosh -> ViewCopy -> GetWorkspaceSize
 * 2. aclnnAcosh: CommonOpExecutorRun
 */

#include "aclnn_acosh.h"
#include "acosh.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

constexpr int64_t ACLNN_MAX_SHAPE_RANK = 8;
// Acosh 支持 Ascend950
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
    }
    // 其他架构暂不支持
    return false;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 输入输出 dtype 必须一致
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    if (!IsDtypeSupported(self->GetDataType())) {
        auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Acosh dtype not supported: dtype=%d, npuArch=%d. "
            "Supported: FLOAT16, FLOAT, BF16 (Ascend950).",
            static_cast<int>(self->GetDataType()), static_cast<int>(npuArch));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    if (IsPrivateFormat(self->GetStorageFormat()) || IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Acosh: private format not supported: self=%d, out=%d",
            static_cast<int>(self->GetStorageFormat()), static_cast<int>(out->GetStorageFormat()));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    if (!CheckNotNull(self, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Acosh: CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(self, out)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Acosh: CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
            static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(self, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Acosh: CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(self, out)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Acosh: CheckShape failed: self_dim=%zu, out_dim=%zu",
            self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 */
extern "C" aclnnStatus aclnnAcoshGetWorkspaceSize(
    const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAcosh, DFX_IN(self), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 Tensor 快速返回
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::Acosh(selfContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnAcosh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAcosh);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
