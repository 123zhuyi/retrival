/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include "aclnn_allto_allv_grouped_mat_mul.h"
#include <algorithm>
#include "allto_allv_grouped_mat_mul_checker.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor *gmmX,
    const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
    const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const aclTensor *gmmXScale,
    const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
    const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
    const aclTensor *mmWeightOffsetOptional, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype,
    int64_t mmDtype, const aclTensor *gmmY, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, bool permuteOutFlag, aclTensor *gmmY, const aclTensor *mmYOptional,
    const aclTensor *permuteOutOptional)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(gmmY, return false);
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, left: %u, right: %u, "
            "mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is nullptr: %u",
            (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))),
            (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))),
            mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output flag does not match optional output ptr!");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const char *group, int64_t epWorldSize, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional)
{
    (void)epWorldSize; // Unused
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
        mmWeightOptional, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional),
        ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(Mc2AlltoAllvGMMChecker::CheckGroup(group), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor *gmmX, const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret_param = CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
        mmWeightOptional, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = Mc2AlltoAllvGMMChecker::CheckSendAndRecv(sendCounts, recvCounts, gmmX, gmmY);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);
    int64_t noQuantMode = 0;
    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();
    int64_t groupSize = 0;
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(gmmX, gmmWeight, sendCountsTensorOptional,
        recvCountsTensorOptional, mmXOptional, mmWeightOptional, 
        nullptr, // gmmXScale
        nullptr,  // gmmWeightScale
        nullptr, // gmmXOffset
        nullptr, // gmmWeightOffset
        nullptr, // mmxScale
        nullptr, // mmWeightScale
        nullptr, // mmxOffset
        nullptr, // mmWeightOffset
        group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag,
        noQuantMode, noQuantMode, noQuantMode, noQuantMode, groupSize, yDtype, mmDtype, gmmY, mmYOptional,
        permuteOutOptional, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif