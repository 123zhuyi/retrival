/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_add_mat_mat_elements_plus.h"
#include "add_mat_mat_elements_plus.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                   DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* c, const aclTensor* a, const aclTensor* b, const aclTensor* beta,
                         const aclTensor* alpha, const aclTensor* cOut)
{
    OP_CHECK_NULL(c, return false);
    OP_CHECK_NULL(a, return false);
    OP_CHECK_NULL(b, return false);
    OP_CHECK_NULL(beta, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(cOut, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* c, const aclTensor* a, const aclTensor* b, const aclTensor* beta,
                            const aclTensor* alpha, const aclTensor* cOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(c, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(a, c->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(b, c->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(beta, c->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(alpha, c->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(cOut, c->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* c, const aclTensor* a, const aclTensor* b, const aclTensor* beta,
                       const aclTensor* alpha, const aclTensor* cOut)
{
    OP_CHECK_MAX_DIM(c, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(a, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(b, MAX_DIM_LEN, return false);

    // beta/alpha 必须是 1-element 标量 tensor（kernel 仅读取第 0 个元素，多元素输入会静默取错值）
    if (beta->GetViewShape().GetShapeSize() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "beta must be a scalar tensor with exactly 1 element.");
        return false;
    }
    if (alpha->GetViewShape().GetShapeSize() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha must be a scalar tensor with exactly 1 element.");
        return false;
    }

    op::Shape broadcastShape;
    if (!BroadcastInferShape(a->GetViewShape(), b->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of a and b can't broadcast.");
        return false;
    }
    if (!BroadcastInferShape(c->GetViewShape(), broadcastShape, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of c and broadcast(a,b) can't broadcast.");
        return false;
    }
    if (broadcastShape != cOut->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape cOut mismatch broadcast result.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* c, const aclTensor* a, const aclTensor* b, const aclTensor* beta,
                               const aclTensor* alpha, const aclTensor* cOut)
{
    CHECK_COND(CheckNotNull(c, a, b, beta, alpha, cOut), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");
    CHECK_COND(CheckDtypeValid(c, a, b, beta, alpha, cOut), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");
    CHECK_COND(CheckShape(c, a, b, beta, alpha, cOut), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddMatMatElementsPlusGetWorkspaceSize(const aclTensor* c, const aclTensor* a, const aclTensor* b,
                                                       const aclTensor* beta, const aclTensor* alpha, aclTensor* cOut,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAddMatMatElementsPlus, DFX_IN(c, a, b, beta, alpha), DFX_OUT(cOut));

    auto ret = CheckParams(c, a, b, beta, alpha, cOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_COND(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR, "CREATE_EXECUTOR failed!");

    if (c->IsEmpty() || a->IsEmpty() || b->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto cContiguous = l0op::Contiguous(c, uniqueExecutor.get());
    CHECK_COND(cContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous c failed!");
    auto aContiguous = l0op::Contiguous(a, uniqueExecutor.get());
    CHECK_COND(aContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous a failed!");
    auto bContiguous = l0op::Contiguous(b, uniqueExecutor.get());
    CHECK_COND(bContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous b failed!");
    auto betaContiguous = l0op::Contiguous(beta, uniqueExecutor.get());
    CHECK_COND(betaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous beta failed!");
    auto alphaContiguous = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_COND(alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous alpha failed!");

    auto result = l0op::AddMatMatElementsPlus(cContiguous, aContiguous, bContiguous, betaContiguous, alphaContiguous,
                                              uniqueExecutor.get());
    CHECK_COND(result != nullptr, ACLNN_ERR_INNER_NULLPTR, "AddMatMatElementsPlus failed!");

    auto viewCopyResult = l0op::ViewCopy(result, cOut, uniqueExecutor.get());
    CHECK_COND(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "ViewCopy failed!");

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAddMatMatElementsPlus(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAddMatMatElementsPlus);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
