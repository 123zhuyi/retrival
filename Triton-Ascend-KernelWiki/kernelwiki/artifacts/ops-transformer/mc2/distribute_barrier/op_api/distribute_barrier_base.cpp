/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file distribute_barrier_base.cpp
 * \brief
 */
#include <algorithm>

#include "aclnn_distribute_barrier_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "distribute_barrier_base.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerDistributeBarrier(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerDistributeBarrierGetWorkspaceSize(const aclTensor* xRef, const aclTensor* timeOut,
                                                               const aclTensor* elasticInfo, const char* group,
                                                               int64_t worldSize, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);

// check nullptr
bool BarrierCheckNullStatus(const aclTensor* xRef, const char* group)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(xRef, return false);
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    return true;
}

// 入参校验
aclnnStatus BarrierCheckParams(const aclTensor* xRef, const char* group)
{
    CHECK_RET(BarrierCheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
    auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
    if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name length in range (0, HCCL_GROUP_NAME_MAX), but it's %zu.",
                groupStrnLen);
        return false;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDistributeBarrierGetWorkspaceSizeBase(const aclTensor* xRef, const aclTensor* timeOut,
                                                       const aclTensor* elasticInfo, const char* group,
                                                       int64_t worldSize, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    auto retParam = BarrierCheckParams(xRef, group);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    return aclnnInnerDistributeBarrierGetWorkspaceSize(xRef, timeOut, elasticInfo,
                                                       group, worldSize, workspaceSize, executor);
}

aclnnStatus aclnnDistributeBarrierBase(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDistributeBarrier(workspace, workspaceSize, executor,
                                       stream);
}
#ifdef __cplusplus
}
#endif
