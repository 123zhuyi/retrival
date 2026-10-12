/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_grouped_matmul_add.h"
#include "aclnn_grouped_matmul_add_v2.h"

#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#include "../../../grouped_matmul/op_host/op_api/grouped_matmul_util.h"
#include "grouped_matmul_add_util.h"
#include "grouped_matmul_add.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static aclnnStatus CheckNotNull(gmm_add_advanced::GroupedMatmulAddParams params)
{
    CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(params.weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight must not be nullptr.");
    CHECK_COND(params.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "groupList must not be nullptr.");
    CHECK_COND(params.yRef != nullptr, ACLNN_ERR_PARAM_NULLPTR, "yRef must not be nullptr.");
    CHECK_COND(params.groupListType == 0 || params.groupListType == 1, ACLNN_ERR_PARAM_INVALID,
               "groupListType shoule be 0 or 1, but actual is: %d", params.groupListType);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(gmm_add_advanced::GroupedMatmulAddParams params)
{
    CHECK_COND(params.x->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of x should be ND, current format is invalid.");
    CHECK_COND(params.weight->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of weight should be ND, current format is invalid.");
    CHECK_COND(params.groupList->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "Format of groupList should be ND, current format is invalid.");
    CHECK_COND(params.yRef->GetStorageFormat() == Format::FORMAT_ND ||
                   params.yRef->GetStorageFormat() == Format::FORMAT_NCL,
               ACLNN_ERR_PARAM_INVALID, "Format of yRef should be ND or NCL, current format is invalid.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(gmm_add_advanced::GroupedMatmulAddParams params)
{
    auto weightDimNum = params.weight->GetViewShape().GetDimNum();
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    auto groupListDimNum = params.groupList->GetViewShape().GetDimNum();
    CHECK_COND(xDimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num
               "The dimension of x should be 2, but actual is %zu.", xDimNum);
    CHECK_COND(weightDimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num
               "The dimension of weight should be 2, but actual is %zu.", weightDimNum);
    CHECK_COND(groupListDimNum == 1, ACLNN_ERR_PARAM_INVALID,
               "The dimension of groupList should be 1, but actual is %ld.", groupListDimNum);
    auto aKDim = params.x->GetViewShape().GetDim(0);
    auto bKDim = params.weight->GetViewShape().GetDim(0);
    auto mDim = params.x->GetViewShape().GetDim(1);

    CHECK_COND(mDim > 0, ACLNN_ERR_PARAM_INVALID, "The M value[%ld] in x should be positive.", mDim);

    CHECK_COND(aKDim == bKDim, ACLNN_ERR_PARAM_INVALID,
               "The kDim of x/weight should be equal, but the actual is %ld/%ld.", aKDim, bKDim);
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        auto groupNum = params.groupList->GetViewShape().GetDim(0);
        auto yDimNum = params.yRef->GetViewShape().GetDimNum();
        CHECK_COND(yDimNum == 3, ACLNN_ERR_PARAM_INVALID, // DAV3510 y tensor need 3 dim
                   "The dimension of y should be 3 , but actual is %zu.", yDimNum);
        auto groupNumY = params.yRef->GetViewShape().GetDim(0);
        CHECK_COND(groupNum == groupNumY, ACLNN_ERR_PARAM_INVALID,
                   "groupNum should be equal between groupListTensor and yTensor when group_type is 2, but actual "
                   "groupListTensor[%ld], yTensor[%ld].",
                   groupNum, groupNumY);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(gmm_add_advanced::GroupedMatmulAddParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x1 failed.");
    CHECK_COND(DataContiguous(params.weight, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x2 failed.");
    CHECK_COND(DataContiguous(params.groupList, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous groupList failed.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(gmm_add_advanced::GroupedMatmulAddParams params)
{
    auto xDtype = params.x->GetDataType();
    auto weightDtype = params.weight->GetDataType();
    CHECK_COND(params.yRef->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "Input yRef dtype should be FLOAT32, actual dtype is %s.",
               op::ToString(params.yRef->GetDataType()).GetString());
    CHECK_COND(params.groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "Input groupList dtype should be INT64, actual dtype is %s.",
               op::ToString(params.groupList->GetDataType()).GetString());
    if ((xDtype != DataType::DT_FLOAT16 && xDtype != DataType::DT_BF16) || (xDtype != weightDtype)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "case with x dtype %s and weight dtype %s is not supported.",
                op::ToString(xDtype).GetString(), op::ToString(weightDtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(gmm_add_advanced::GroupedMatmulAddParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSizeCommon(gmm_add_advanced::GroupedMatmulAddParams params,
                                                               uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    auto result = l0op::GroupedMatmulAdd(params.x, params.weight, params.groupList, params.yRef, params.transposeX,
                                         params.transposeWeight, params.groupType, params.groupListType, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result, params.yRef, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                  const aclTensor *groupList, aclTensor *yRef, bool transposeX,
                                                  bool transposeWeight, int64_t groupType, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor)
{
    gmm_add_advanced::GroupedMatmulAddParams params{x, weight, groupList, yRef, transposeX, transposeWeight, groupType};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnGroupedMatmulAdd, DFX_IN(x, weight, groupList, yRef, transposeX, transposeWeight, groupType),
                   DFX_OUT(yRef));
    return aclnnGroupedMatmulAddGetWorkspaceSizeCommon(params, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                    const aclTensor *groupList, aclTensor *yRef, bool transposeX,
                                                    bool transposeWeight, int64_t groupType, int64_t groupListType,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    gmm_add_advanced::GroupedMatmulAddParams params{x,          weight,          groupList, yRef,
                                                    transposeX, transposeWeight, groupType, groupListType};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnGroupedMatmulAddV2,
                   DFX_IN(x, weight, groupList, yRef, transposeX, transposeWeight, groupType, groupListType),
                   DFX_OUT(yRef));
    return aclnnGroupedMatmulAddGetWorkspaceSizeCommon(params, workspaceSize, executor);
}

aclnnStatus aclnnGroupedMatmulAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulAdd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in QuantGMMInplaceAdd launch aicore.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulAddV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulAddV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in QuantGMMInplaceAdd launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
