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
 * @file aclnn_atan_grad.cpp
 * @brief ACLNN L2 API 实现 - AtanGrad 算子
 *
 * 两段式接口实现：
 *   第一段：aclnnAtanGradGetWorkspaceSize
 *     CREATE_EXECUTOR -> CheckParams -> HasEmptyTensor -> Contiguous
 *     -> l0op::AtanGrad -> ViewCopy -> GetWorkspaceSize
 *   第二段：aclnnAtanGrad
 *     CommonOpExecutorRun
 */

#include "aclnn_atan_grad.h"
#include "atan_grad.h"
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

static const std::initializer_list<op::DataType> ATAN_GRAD_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* x, const aclTensor* dy, const aclTensor* dx)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(dx, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* dy, const aclTensor* dx)
{
    // x、dy、dx 三者 dtype 必须相同
    OP_CHECK_DTYPE_NOT_MATCH(dy, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(dx, x->GetDataType(), return false);

    if (!CheckType(x->GetDataType(), ATAN_GRAD_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AtanGrad: unsupported dtype=%d. Supported: FLOAT16, FLOAT, BF16.",
            static_cast<int>(x->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* dy, const aclTensor* dx)
{
    if (IsPrivateFormat(x->GetStorageFormat()) || IsPrivateFormat(dy->GetStorageFormat()) ||
        IsPrivateFormat(dx->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "AtanGrad: private format not supported: x=%d, dy=%d, dx=%d",
            static_cast<int>(x->GetStorageFormat()), static_cast<int>(dy->GetStorageFormat()),
            static_cast<int>(dx->GetStorageFormat()));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* dy, const aclTensor* dx)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(dy, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(dx, ACLNN_MAX_SHAPE_RANK, return false);

    // x.shape 与 dy.shape 必须完全相同，否则存在 GM 越界访问风险
    if (x->GetViewShape() != dy->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AtanGrad: x.shape must equal dy.shape, but got different shapes.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* dy, const aclTensor* dx)
{
    if (!CheckNotNull(x, dy, dx)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(x, dy, dx)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(x, dy, dx)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(x, dy, dx)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static bool HasEmptyTensor(const aclTensor* x, const aclTensor* dy) { return x->IsEmpty() || dy->IsEmpty(); }

extern "C" aclnnStatus aclnnAtanGradGetWorkspaceSize(
    const aclTensor* x, const aclTensor* dy, const aclTensor* dx, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAtanGrad, DFX_IN(x, dy), DFX_OUT(dx));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, dy, dx);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x, dy)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dyContiguous = l0op::Contiguous(dy, uniqueExecutor.get());
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::AtanGrad(xContiguous, dyContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, dx, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnAtanGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAtanGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
