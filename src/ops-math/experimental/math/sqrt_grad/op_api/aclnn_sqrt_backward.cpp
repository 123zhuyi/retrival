/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sqrt_backward.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "sqrt_grad.h"
#include "aclnn_util.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> SUPPORTED_DTYPES = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *y, const aclTensor *dy, const aclTensor *z)
{
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(dy, return false);
    OP_CHECK_NULL(z, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *y, const aclTensor *dy, const aclTensor *z)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(y, SUPPORTED_DTYPES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dy, SUPPORTED_DTYPES, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(z, SUPPORTED_DTYPES, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, dy->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, z->GetDataType(), return false);
    return true;
}

static bool CheckShapeValid(const aclTensor *y, const aclTensor *dy, const aclTensor *z)
{
    OP_CHECK_SHAPE_NOT_EQUAL(y, dy, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(y, z, return false);
    OP_CHECK_MAX_DIM(y, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(dy, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(z, MAX_DIM_LEN, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *y, const aclTensor *dy, const aclTensor *z)
{
    CHECK_RET(CheckNotNull(y, dy, z), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(y, dy, z), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(y, dy, z), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus BuildExecutor(const aclTensor *y, const aclTensor *dy, aclTensor *z, aclOpExecutor **executor,
                                 uint64_t *workspaceSize)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dyContiguous = l0op::Contiguous(dy, uniqueExecutor.get());
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sqrtGradOut = l0op::SqrtGrad(yContiguous, dyContiguous, uniqueExecutor.get());
    CHECK_RET(sqrtGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(sqrtGradOut, z, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSqrtBackwardGetWorkspaceSize(const aclTensor *y, const aclTensor *dy, aclTensor *z,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret = CheckParams(y, dy, z);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (y->IsEmpty()) {
        *workspaceSize = 0;
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    return BuildExecutor(y, dy, z, executor, workspaceSize);
}

aclnnStatus aclnnSqrtBackwardGetWorkspaceSize(
    const aclTensor *y, const aclTensor *dy, aclTensor *z, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnSqrtBackward, DFX_IN(y, dy), DFX_OUT(z));
    return ExecSqrtBackwardGetWorkspaceSize(y, dy, z, workspaceSize, executor);
}

aclnnStatus aclnnSqrtBackward(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSqrtBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
