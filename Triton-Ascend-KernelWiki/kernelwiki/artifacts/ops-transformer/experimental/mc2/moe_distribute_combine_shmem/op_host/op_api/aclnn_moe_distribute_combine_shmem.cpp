/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_moe_distribute_combine_shmem.h"
#include "op_mc2.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
  NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
  NNOPBASE_HCCL_SERVER_TYPE_MTE,
  NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerMoeDistributeCombineShmemGetWorkspaceSize(
    const aclTensor* shmemSpace, const aclTensor* expandX,
    const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts, const aclTensor* expertScales,
    const aclTensor* tpSendCounts, const aclTensor* xActiveMask,
    const aclTensor* activationScale, const aclTensor* weightScale,
    const aclTensor* groupList, const aclTensor* expandScales,
    const aclTensor* sharedExpertX, const aclTensor* elasticInfo,
    const aclTensor* oriX, const aclTensor* constExpertAlpha1,
    const aclTensor* constExpertAlpha2, const aclTensor* constExpertV,
    const char* groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    int64_t shmem_size, aclTensor* x, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeDistributeCombineShmem(void* workspace,
                                                    uint64_t workspaceSize,
                                                    aclOpExecutor* executor,
                                                    aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(
    void* executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNotNull(const aclTensor* expandX, const aclTensor* expertIds,
                         const aclTensor* assistInfoForCombine,
                         const aclTensor* epSendCounts,
                         const aclTensor* expertScales, const char* groupEp,
                         aclTensor* x) {
  OP_LOGD("aclnn_moe_distribute_combine_shmem CheckNotNull start");
  OP_CHECK_NULL(expandX, return false);
  OP_CHECK_NULL(expertIds, return false);
  OP_CHECK_NULL(assistInfoForCombine, return false);
  OP_CHECK_NULL(epSendCounts, return false);
  OP_CHECK_NULL(expertScales, return false);
  OP_CHECK_NULL(x, return false);
  OP_LOGD("aclnn_moe_distribute_combine_shmem CheckNotNull success");
  return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* expandX,
                               const aclTensor* expertIds,
                               const aclTensor* expandIdx,
                               const aclTensor* epSendCounts,
                               const aclTensor* expertScales,
                               const char* groupEp, const char* groupTp,
                               aclTensor* x) {
  OP_LOGD("aclnn_moe_distribute_combine_shmem checkparams start");
  CHECK_RET(CheckNotNull(expandX, expertIds, expandIdx, epSendCounts,
                         expertScales, groupEp, x),
            ACLNN_ERR_PARAM_NULLPTR);

  if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupEp name exceeds %zu.",
            HCCL_GROUP_NAME_MAX);
    return ACLNN_ERR_PARAM_INVALID;
  }
  if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required groupTp name exceeds %zu.",
            HCCL_GROUP_NAME_MAX);
    return ACLNN_ERR_PARAM_INVALID;
  }
  OP_LOGD("aclnn_moe_distribute_combine_shmem checkparams success");
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeDistributeCombineShmemGetWorkspaceSize(
    const aclTensor* shmemSpace, const aclTensor* expandX,
    const aclTensor* expertIds, const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts, const aclTensor* expertScales,
    const aclTensor* tpSendCountsOptional, const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional,
    const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional,
    const aclTensor* constExpertAlpha2Optional,
    const aclTensor* constExpertVOptional, const char* groupEp,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    const char* groupTp, int64_t tpWorldSize, int64_t tpRankId,
    int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
    int64_t commQuantMode, int64_t groupListType, const char* commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    int64_t shmem_size, aclTensor* xOut, uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  return aclnnInnerMoeDistributeCombineShmemGetWorkspaceSize(
      shmemSpace, expandX, expertIds, assistInfoForCombine, epSendCounts,
      expertScales, tpSendCountsOptional, xActiveMaskOptional,
      activationScaleOptional, weightScaleOptional, groupListOptional,
      expandScalesOptional, sharedExpertXOptional, elasticInfoOptional,
      oriXOptional, constExpertAlpha1Optional, constExpertAlpha2Optional,
      constExpertVOptional, groupEp, epWorldSize, epRankId, moeExpertNum,
      groupTp, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
      sharedExpertRankNum, globalBs, outDtype, commQuantMode, groupListType,
      commAlg, zeroExpertNum, copyExpertNum, constExpertNum, shmem_size, xOut,
      workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeCombineShmem(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor,
                                        aclrtStream stream) {
  return aclnnInnerMoeDistributeCombineShmem(workspace, workspaceSize, executor,
                                          stream);
}

#ifdef __cplusplus
}
#endif