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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "aclnn_approximate_equal.h"
#include "approximate_equal.h"

#include <cmath>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,
    DataType::DT_FLOAT16,
    DataType::DT_BF16,
};

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST); }

static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* y)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* y)
{
    // x1 dtype must be supported; x2 must match x1; y must be BOOL.
    if (!IsDtypeSupported(x1->GetDataType())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "ApproximateEqual: unsupported x1 dtype=%d (allowed: FLOAT / FLOAT16 / BF16).",
            static_cast<int>(x1->GetDataType()));
        return false;
    }
    if (x1->GetDataType() != x2->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "ApproximateEqual: x1 dtype (%d) != x2 dtype (%d).",
            static_cast<int>(x1->GetDataType()), static_cast<int>(x2->GetDataType()));
        return false;
    }
    if (y->GetDataType() != DataType::DT_BOOL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "ApproximateEqual: y dtype must be BOOL, got %d.",
            static_cast<int>(y->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* y)
{
    if (IsPrivateFormat(x1->GetStorageFormat()) || IsPrivateFormat(x2->GetStorageFormat()) ||
        IsPrivateFormat(y->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ApproximateEqual: private formats not supported.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x1, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(x2, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);
    if (x1->GetViewShape() != x2->GetViewShape() || x1->GetViewShape() != y->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ApproximateEqual: shape mismatch among x1/x2/y.");
        return false;
    }
    return true;
}

static bool CheckTolerance(float tolerance)
{
    if (std::isnan(tolerance) || std::isinf(tolerance) || tolerance < 0.0f) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "ApproximateEqual: tolerance=%f is illegal (must be finite and >=0).", tolerance);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* y, float tolerance)
{
    if (!CheckNotNull(x1, x2, y)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x1, x2, y) || !CheckFormat(x1, x2, y) || !CheckShape(x1, x2, y) ||
        !CheckTolerance(tolerance)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApproximateEqualGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, float tolerance, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnApproximateEqual, DFX_IN(x1, x2, tolerance), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x1, x2, y, tolerance);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Empty tensor -> no-op.
    if (x1->IsEmpty() || x2->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::ApproximateEqual(x1Contiguous, x2Contiguous, tolerance, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnApproximateEqual(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnApproximateEqual);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
