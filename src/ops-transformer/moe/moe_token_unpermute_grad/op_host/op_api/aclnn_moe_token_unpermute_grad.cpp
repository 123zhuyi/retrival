/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_moe_token_unpermute_grad.cpp
 * \brief
 */
#include "aclnn_moe_token_unpermute_grad.h"
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

extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* expandedRowIdx, const aclTensor* expandedXOptional,
    const aclTensor* scalesOptional, const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
    int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity,
    const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
extern aclnnStatus aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
    const aclTensor* permuteTokens, const aclTensor* unpermutedTokensGrad, const aclTensor* sortedIndices,
    const aclTensor* probsOptional, bool paddedMode, const aclIntArray* restoreShapeOptional,
    aclTensor* permutedTokensGradOut, aclTensor* probsGradOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    static bool useMoeFinalizeRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGradGetWorkspaceSize(
            permuteTokens, unpermutedTokensGrad, sortedIndices, probsOptional, paddedMode, restoreShapeOptional,
            permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(permuteTokens, return ACLNN_ERR_PARAM_NULLPTR);
    auto permuteTokensShape = permuteTokens->GetViewShape();
    CHECK_RET(permuteTokensShape.GetDimNum() > 0, ACLNN_ERR_PARAM_INVALID);
    int64_t activeNum = permuteTokensShape.GetDim(0);
    aclnnStatus ret = aclnnInnerMoeFinalizeRoutingV2GradGetWorkspaceSize(
        unpermutedTokensGrad, sortedIndices, permuteTokens, probsOptional, nullptr, nullptr, 0, activeNum, 0, 0,
        permutedTokensGradOut, probsGradOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokenUnpermuteGrad calls aclnnInnerMoeFinalizeRoutingV2Grad, please refer to the document for "
            "parameter correspondence.");
    }
    return ret;
}

aclnnStatus aclnnMoeTokenUnpermuteGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool useMoeFinalizeRoutingV2Grad = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!useMoeFinalizeRoutingV2Grad) {
        return aclnnInnerMoeTokenUnpermuteGrad(workspace, workspaceSize, executor, stream);
    }
    aclnnStatus ret = aclnnInnerMoeFinalizeRoutingV2Grad(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER,
            "aclnnMoeTokenUnpermuteGrad calls aclnnInnerMoeFinalizeRoutingV2Grad, please refer to the document for "
            "parameter correspondence.");
        return ret;
    }
}

#ifdef __cplusplus
}
#endif