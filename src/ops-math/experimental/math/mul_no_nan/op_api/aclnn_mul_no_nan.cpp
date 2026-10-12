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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_mul_no_nan.cpp
 * @brief ACLNN L2 API implementation for MulNoNan
 */

#include "aclnn_mul_no_nan.h"
#include "mul_no_nan.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include <algorithm>

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST); }

static bool HasEmptyTensor(const aclTensor* x, const aclTensor* y) { return x->IsEmpty() || y->IsEmpty(); }

static bool CheckNotNull(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(z, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    OP_CHECK_DTYPE_NOT_MATCH(x, y->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(z, x->GetDataType(), return false);

    OP_CHECK(
        IsDtypeSupported(x->GetDataType()),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dtype not supported: dtype=%d. Supported: FLOAT16, FLOAT, BFLOAT16.",
            static_cast<int>(x->GetDataType())),
        return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    auto formatX = x->GetStorageFormat();
    auto formatY = y->GetStorageFormat();
    auto formatZ = z->GetStorageFormat();

    OP_CHECK(
        !(IsPrivateFormat(formatX) || IsPrivateFormat(formatY) || IsPrivateFormat(formatZ)),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Private format not supported: x=%d, y=%d, z=%d", static_cast<int>(formatX),
            static_cast<int>(formatY), static_cast<int>(formatZ)),
        return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(z, ACLNN_MAX_SHAPE_RANK, return false);

    // Verify x and y shapes are broadcast-compatible
    auto xShape = x->GetViewShape();
    auto yShape = y->GetViewShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t yDimNum = yShape.GetDimNum();
    size_t maxDimNum = std::max(xDimNum, yDimNum);

    for (size_t i = 0; i < maxDimNum; ++i) {
        int64_t xDim = (i < xDimNum) ? xShape.GetDim(xDimNum - 1 - i) : 1;
        int64_t yDim = (i < yDimNum) ? yShape.GetDim(yDimNum - 1 - i) : 1;
        if (xDim != yDim && xDim != 1 && yDim != 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Shapes are not broadcast-compatible at dim %zu (from right): "
                "x_dim=%ld, y_dim=%ld",
                i, xDim, yDim);
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* y, const aclTensor* z)
{
    CHECK_COND(CheckNotNull(x, y, z), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(
        CheckDtypeValid(x, y, z), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed: x_dtype=%d, y_dtype=%d, z_dtype=%d",
        static_cast<int>(x->GetDataType()), static_cast<int>(y->GetDataType()), static_cast<int>(z->GetDataType()));
    CHECK_COND(
        CheckFormat(x, y, z), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed: x_format=%d, y_format=%d, z_format=%d",
        static_cast<int>(x->GetStorageFormat()), static_cast<int>(y->GetStorageFormat()),
        static_cast<int>(z->GetStorageFormat()));
    CHECK_COND(
        CheckShape(x, y, z), ACLNN_ERR_PARAM_INVALID, "CheckShape failed: x_dim=%zu, y_dim=%zu, z_dim=%zu",
        x->GetViewShape().GetDimNum(), y->GetViewShape().GetDimNum(), z->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnMulNoNanGetWorkspaceSize(
    const aclTensor* x, const aclTensor* y, const aclTensor* z, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMulNoNan, DFX_IN(x, y), DFX_OUT(z));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, y, z);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x, y)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::MulNoNan(xContiguous, yContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, z, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnMulNoNan(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMulNoNan);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
