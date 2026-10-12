/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CHANNEL_SHUFFLE_H_
#define OP_API_INC_LEVEL2_ACLNN_CHANNEL_SHUFFLE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnChannelShuffle的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将$(*, C, H, W)$张量的channels分成$g$个组，然后重新排列成$(*, C_{,}^{\underline{g}}g, H, W)$，
 * 同时保持最终输出张量的shape和输入张量保持一致。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持BFLOAT16、FLOAT16、FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、
 * LONG、COMPLEX64、COMPLEX128、BOOL，支持非连续的Tensor，数据格式支持ND，数据维度大于2且不支持7维以上。
 * @param [in] groups：host侧的int64_t， 表示将输入self的channels分成多少组，值需要大于0且要能被self的channels整除。
 * @param [in] out: npu device侧的aclTensor，数据类型和数据维度需要与self一致，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnChannelShuffleGetWorkspaceSize(const aclTensor* self, int64_t groups, aclTensor* out,
                                                          uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnChannelShuffle的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnChannelShuffleGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnChannelShuffle(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_CHANNEL_SHUFFLE_H_
