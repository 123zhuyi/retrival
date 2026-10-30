/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LogicalXor_H_
#define OP_API_INC_LogicalXor_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLogicalXor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成给定输入张量元素的逻辑异或运算。当self和other为非bool类型时，0被视为False，非0被视为True。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalXor])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalXor])
 * D([LogicalXor])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且shape需要与other满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与other一致。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且shape需要是self与other broadcast之后的shape，数据格式支持ND,
 * 且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLogicalXorGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLogicalXor的第二段接口，用于执行计算。
 *
 * 算子功能：完成给定输入张量元素的逻辑异或运算。当self和other为非bool类型时，0被视为False，非0被视为True。
 *
 * 实现说明：
 * api计算的基本路径:
 * ```mermaid
 * graph LR
 * A1[(self)] -->B1([Contiguous])-->C1([Cast])-->D([LogicalXor])
 * A2[(other)]-->B2([Contiguous])-->C2([Cast])-->D([LogicalXor])
 * D([LogicalXor])-->E([Cast])-->F([ViewCopy])-->G[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnLogicalXorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLogicalXor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LogicalXor_H_