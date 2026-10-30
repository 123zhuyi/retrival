/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_acos_grad.cpp
 * @brief ACLNN L2 API 实现 - AcosGrad 算子
 */

#include "aclnn_acos_grad.h"
#include "acos_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return CheckType(dtype, ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
    }
    if (IsRegBase()) {
        return CheckType(dtype, ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static bool HasEmptyTensor(const aclTensor* y_grad, const aclTensor* x) { return y_grad->IsEmpty() || x->IsEmpty(); }

static bool CheckNotNull(const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad)
{
    OP_CHECK_NULL(y_grad, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(x_grad, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad)
{
    OP_CHECK_DTYPE_NOT_MATCH(y_grad, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(x_grad, y_grad->GetDataType(), return false);

    if (!IsDtypeSupported(y_grad->GetDataType())) {
        auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "AcosGrad: Dtype not supported: dtype=%d, npuArch=%d. "
            "Supported: FLOAT16, FLOAT32, BF16.",
            static_cast<int>(y_grad->GetDataType()), static_cast<int>(npuArch));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad)
{
    auto fmtYGrad = y_grad->GetStorageFormat();
    auto fmtX = x->GetStorageFormat();
    auto fmtXGrad = x_grad->GetStorageFormat();

    if (IsPrivateFormat(fmtYGrad) || IsPrivateFormat(fmtX) || IsPrivateFormat(fmtXGrad)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AcosGrad: Private format not supported: y_grad=%d, x=%d, x_grad=%d",
            static_cast<int>(fmtYGrad), static_cast<int>(fmtX), static_cast<int>(fmtXGrad));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad)
{
    OP_CHECK_MAX_DIM(y_grad, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(x_grad, ACLNN_MAX_SHAPE_RANK, return false);

    auto yGradShape = y_grad->GetViewShape();
    auto xShape = x->GetViewShape();
    auto xGradShape = x_grad->GetViewShape();

    if (yGradShape != xShape || yGradShape != xGradShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AcosGrad: Shape mismatch: y_grad=%s, x=%s, x_grad=%s",
            op::ToString(yGradShape).GetString(), op::ToString(xShape).GetString(),
            op::ToString(xGradShape).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad)
{
    if (!CheckNotNull(y_grad, x, x_grad)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AcosGrad: CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(y_grad, x, x_grad)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AcosGrad: CheckDtypeValid failed: y_grad_dtype=%d, x_dtype=%d, x_grad_dtype=%d",
            static_cast<int>(y_grad->GetDataType()), static_cast<int>(x->GetDataType()),
            static_cast<int>(x_grad->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(y_grad, x, x_grad)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGrad: CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(y_grad, x, x_grad)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AcosGrad: CheckShape failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAcosGradGetWorkspaceSize(
    const aclTensor* y_grad, const aclTensor* x, const aclTensor* x_grad, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAcosGrad, DFX_IN(y_grad, x), DFX_OUT(x_grad));

    OP_CHECK_NULL(workspaceSize, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(executor, return ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(y_grad, x, x_grad);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(y_grad, x)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto yGradContiguous = l0op::Contiguous(y_grad, uniqueExecutor.get());
    CHECK_RET(yGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::AcosGrad(yGradContiguous, xContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, x_grad, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAcosGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAcosGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
