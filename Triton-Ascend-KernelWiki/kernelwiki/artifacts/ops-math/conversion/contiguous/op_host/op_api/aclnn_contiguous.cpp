/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnContiguousGetWorkspaceSize(
    const aclTensorList* self, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnContiguous, DFX_IN(self), DFX_OUT());
    CHECK_NOT_NULL(self, workspaceSize, executor);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    uniqueExecutor->AbandonCache();

    // 调用非连续到连续转换
    const uint64_t count = self->Size();
    FVector<const aclTensor*> out(count, nullptr);
    FVector<uint64_t> workspaceOffsets(count, 0U);
    for (uint64_t i = 0U; i < count; ++i) {
        out[i] = l0op::Contiguous((*self)[i], uniqueExecutor.get());
        CHECK_RET(out[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetLinearWorkspaceSize();

    // set offset to exe
    for (uint64_t i = 0U; i < count; ++i) {
        workspaceOffsets[i] = out[i]->GetWorkspaceOffset(); // has check out[i] not null
    }
    uniqueExecutor->SetWorkspaceOffsets(workspaceOffsets);
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnContiguous(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnContiguous);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnViewCopyGetWorkspaceSize(
    const aclTensorList* in, const aclTensorList* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnViewCopy, DFX_IN(in), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    uniqueExecutor->AbandonCache();
    CHECK_RET(in != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用viewcopy
    const uint64_t count = in->Size();
    CHECK_RET(count == out->Size(), ACLNN_ERR_PARAM_INVALID);
    for (uint64_t i = 0U; i < count; ++i) {
        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy((*in)[i], (*out)[i], uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetLinearWorkspaceSize();

    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnViewCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnViewCopy);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
