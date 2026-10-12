/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_NORMAL_OUT_H_
#define OP_API_INC_NORMAL_OUT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnNormal的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 * @param [in] mean: npu
 * device侧的aclTensor，数据类型支持浮点数据类型，且数据类型需要与std构成互相推导关系，shape需要与std满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与std一致。
 * @param [in] std: npu
 * device侧的aclTensor，数据类型支持浮点类型，且数据类型需要与mean构成互相推导关系，shape需要与mean满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与mean一致。
 * @param [in] seed: host侧的int64_t，伪随机数生成器的种子值。
 * @param [in] offset: host侧的int64_t， 伪随机数生成器的偏移量。
 * @param [out] out: host侧的int64_t, 输出张量。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(
    const aclTensor* mean, const aclTensor* std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnNormalTensorFloat的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnNormalTensorFloatGetWorkspaceSize(
    const aclTensor* mean, float std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnNormalFloatTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnNormalFloatTensorGetWorkspaceSize(
    float mean, const aclTensor* std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnNormalFloatFloat的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnNormalFloatFloatGetWorkspaceSize(
    float mean, float std, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnNormal的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnNormalTensorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnNormalTensorFloat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnNormalFloatTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnNormalFloatFloat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_NORMAL_OUT_H_
