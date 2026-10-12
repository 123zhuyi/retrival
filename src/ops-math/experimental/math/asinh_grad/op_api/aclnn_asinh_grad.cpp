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

/**
 * @file aclnn_asinh_grad.cpp
 * @brief ACLNN L2 API implementation for AsinhGrad
 *
 * Two-phase design:
 * 1. aclnnAsinhGradGetWorkspaceSize - parameter check, contiguous, create executor
 * 2. aclnnAsinhGrad - execute computation
 */

#include "aclnn_asinh_grad.h"
#include "asinh_grad.h"
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

// Iteration 3: FP32 + FP16 + BF16
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST); }

static bool HasEmptyTensor(const aclTensor* y, const aclTensor* dy) { return y->IsEmpty() || dy->IsEmpty(); }

static bool CheckNotNull(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(z, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    // y, dy, z must have the same dtype
    OP_CHECK_DTYPE_NOT_MATCH(y, dy->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(z, y->GetDataType(), return false);

    if (!IsDtypeSupported(y->GetDataType())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AsinhGrad: unsupported dtype=%d. Supported: FLOAT, FLOAT16, BF16.",
            static_cast<int>(y->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    auto formatY = y->GetStorageFormat();
    auto formatDy = dy->GetStorageFormat();
    auto formatZ = z->GetStorageFormat();

    if (IsPrivateFormat(formatY) || IsPrivateFormat(formatDy) || IsPrivateFormat(formatZ)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AsinhGrad: private format not supported: y=%d, dy=%d, z=%d",
            static_cast<int>(formatY), static_cast<int>(formatDy), static_cast<int>(formatZ));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(dy, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(z, ACLNN_MAX_SHAPE_RANK, return false);

    // y and dy must have the same shape (no broadcast supported)
    if (y->GetViewShape() != dy->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AsinhGrad: y and dy shape mismatch (broadcast not supported)");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL(z, y, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* y, const aclTensor* dy, const aclTensor* z)
{
    if (!CheckNotNull(y, dy, z)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "AsinhGrad: null tensor pointer");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(y, dy, z)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AsinhGrad: dtype check failed: y=%d, dy=%d, z=%d",
            static_cast<int>(y->GetDataType()), static_cast<int>(dy->GetDataType()),
            static_cast<int>(z->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(y, dy, z)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(y, dy, z)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AsinhGrad: shape check failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAsinhGradGetWorkspaceSize(
    const aclTensor* y, const aclTensor* dy, aclTensor* z, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAsinhGrad, DFX_IN(y, dy), DFX_OUT(z));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(y, dy, z);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(y, dy)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dyContiguous = l0op::Contiguous(dy, uniqueExecutor.get());
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::AsinhGrad(yContiguous, dyContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, z, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAsinhGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAsinhGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
