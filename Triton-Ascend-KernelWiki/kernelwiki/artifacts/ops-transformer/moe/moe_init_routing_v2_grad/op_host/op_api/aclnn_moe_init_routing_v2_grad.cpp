/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_init_routing_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
    int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
    int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnInnerMoeInitRoutingV2GradGetWorkspaceSize(
        gradExpandedX, expandedRowIdx, topK, dropPadMode, activeNum, out, workspaceSize, executor);
}

aclnnStatus aclnnMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerMoeInitRoutingV2Grad(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif