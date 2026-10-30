/**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_DOT_H_
#define OP_API_INC_LEVEL2_ACLNN_DOT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：计算两个一维张量的点积结果
 * 计算公式：如下
 * $$
 * self = [x_{1}, x_{2}, ..., x_{n}]
 * $$
 *
 * $$
 * tensor = [y_{1}, y_{2}, ..., y_{n}]
 * $$
 *
 * $$
 * out = x_{1}*y_{1} + x_{2}*y_{2} + ... + x_{n}*y_{n}
 * $$
 *
 * 实现说明：
 * api计算的基本路径一（self和tensor均不是空Tensor场景）：
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::Dot])
 *     D[(tensor)] --> E([l0op::Contiguous])
 *     E --> C
 *     C --> F([l0op::ViewCopy])
 *     F --> G[(out)]
 * ```
 *
 * api计算的基本路径二（self和tensor均为空Tensor场景）：
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Fill])
 *     C[(tensor)] --> B
 *     B --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 */

/**
 * @brief aclnnDot的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT8、INT32、UINT8，且数据类型需要与tensor一致，
 * 支持非连续的Tensor，数据格式支持ND，维度是1维，且shape需要与tensor一致。
 * @param [in] tensor: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT8、INT32、UINT8，且数据类型需要与self一致，
 * 支持非连续的Tensor，数据格式支持ND，维度是1维，且shape需要与self一致。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT8、INT32、UINT8，且数据类型需要与self、tensor
 * 一致，数据格式支持ND，维度是0维。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDotGetWorkspaceSize(const aclTensor* self, const aclTensor* tensor, aclTensor* out,
                                               uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDot的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnDotGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_DOT_H_
