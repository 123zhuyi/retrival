/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aclnn_moe_finalize_routing_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

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

aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* expandedRowIdx, const aclTensor* expandedXOptional,
    const aclTensor* scalesOptional, const aclTensor* expertIdxOptional, const aclTensor* biasOptional,
    int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity,
    const aclTensor* gradExpandedXOut, const aclTensor* gradScalesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnInnerMoeFinalizeRoutingV2GradGetWorkspaceSize(
        gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional, dropPadMode,
        activeNum, expertNum, expertCapacity, gradExpandedXOut, gradScalesOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeFinalizeRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerMoeFinalizeRoutingV2Grad(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif