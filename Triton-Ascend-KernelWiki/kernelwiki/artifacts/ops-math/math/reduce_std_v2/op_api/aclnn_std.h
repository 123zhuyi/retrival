/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_STD_H_
#define OP_API_INC_STD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnStd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算样本标准差
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     D([dim]) --> C([l0op::ReduceMean])
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C
 *     E([keepdim=true]) --> C
 *     C --> H([l0op::Expand])
 *     J([dim]) --> I
 *     K([correction]) --> I
 *     O[(self)] --> I
 *     P --> I([l0op::ReduceStdWithMean])
 *     L([keepdim]) --> I
 *     H --> P[(meanOut)]
 *     I --> M([l0op::ViewCopy])
 *     M --> N[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。数据格式支持ND。支持非连续的Tensor。
 * @param [in] dim: host侧的aclIntArray，支持的数据类型为INT32、INT64。
 * @param [in] correction: host侧的整形，修正值。
 * @param [in] keepdim: host侧的布尔型，是否在输出张量中保留输入张量的维度。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。数据格式支持ND。支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnStdGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, const int64_t correction,
                                               bool keepdim, aclTensor* out, uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief aclnnStd的第二段接口，用于执行计算。
 *
 * 算子功能：计算样本标准差
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     D([dim]) --> C([l0op::ReduceMean])
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C
 *     E([keepdim=true]) --> C
 *     C --> H([l0op::Expand])
 *     J([dim]) --> I
 *     K([correction]) --> I
 *     O[(self)] --> I
 *     P --> I([l0op::ReduceStdWithMean])
 *     L([keepdim]) --> I
 *     H --> P[(meanOut)]
 *     I --> M([l0op::ViewCopy])
 *     M --> N[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSubGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnStd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                               const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_STD_H_
