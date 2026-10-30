/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_permute_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_LOG_RET(cond, log_func, return_expr) \
    do {                                           \
        if (cond) {                                \
            log_func;                              \
        }                                          \
        return_expr;                               \
    } while (0)

extern aclnnStatus aclnnInnerMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
    int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

extern aclnnStatus aclnnInnerMoeTokenPermuteGradGetWorkspaceSize(
    const aclTensor* permutedOutputGrad, const aclTensor* sortedIndices, int64_t numTopk, bool paddedMode,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenPermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(
    const aclTensor* permutedOutputGrad, const aclTensor* sortedIndices, int64_t numTopk, bool paddedMode,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    static bool useMoeInitRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2Grad) {
        return aclnnInnerMoeTokenPermuteGradGetWorkspaceSize(
            permutedOutputGrad, sortedIndices, numTopk, paddedMode, out, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(permutedOutputGrad, return ACLNN_ERR_PARAM_NULLPTR);
    auto permutedOutputGradShape = permutedOutputGrad->GetViewShape();
    CHECK_RET(permutedOutputGradShape.GetDimNum() > 0, ACLNN_ERR_PARAM_INVALID);
    int64_t activeNum = permutedOutputGradShape.GetDim(0);
    aclnnStatus ret = aclnnInnerMoeInitRoutingV2GradGetWorkspaceSize(
        permutedOutputGrad, sortedIndices, numTopk, 0, activeNum, out, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokenPermuteGrad calls aclnnInnerMoeInitRoutingV2Grad, please refer to the document for "
            "parameter correspondence.");
    }
    return ret;
}

aclnnStatus aclnnMoeTokenPermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeInitRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeInitRoutingV2Grad) {
        return aclnnInnerMoeTokenPermuteGrad(workspace, workspaceSize, executor, stream);
    }
    aclnnStatus ret = aclnnInnerMoeInitRoutingV2Grad(workspace, workspaceSize, executor, stream);
    CHECK_LOG_RET(
        ret != ACLNN_SUCCESS,
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokenPermuteGrad calls aclnnInnerMoeInitRoutingV2Grad, please refer to the document for "
            "parameter correspondence."),
        return ret);
}

#ifdef __cplusplus
}
#endif