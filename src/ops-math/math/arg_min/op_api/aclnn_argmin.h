/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_ARGMIN_H_
#define OP_API_INC_ARGMIN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnARGMIN的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：返回张量在指定维度上的最小值的索引。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> C([l0op::ArgMin])
 *  C --> F([l0op::Cast])
 *  D([dim]) --> C
 *  F -.-> E([l0op::ViewCopy])
 *  E --> O[(Out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、FLOAT64、INT8、
 * INT16、INT32、INT64、UINT8、BFLOAT16，数据格式支持ND。支持非连续的Tensor。
 * @param [in] dim: host侧int64类型，指定了要进行最小值计算的维度。
 * @param [in] keepdim: host侧的布尔型，是否在输出张量中保留输入张量的维度。
 * @param [in] out: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnArgMinGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim,
                                                  aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnArgMin的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * 算子功能：返回张量在指定维度上的最大值的索引。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *  A[(self)] -.->B([l0op::Contiguous])
 *  B --> C([l0op::ArgMin])
 *  C --> F([l0op::Cast])
 *  D([dim]) --> C
 *  F -.-> E([l0op::ViewCopy])
 *  E --> O[(Out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnArgMinGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnArgMin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_ARGMIN_H_