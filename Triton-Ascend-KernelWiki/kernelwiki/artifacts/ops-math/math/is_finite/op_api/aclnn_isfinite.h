/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ISFINITE_H_
#define OP_API_INC_LEVEL2_ISFINITE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：判断输入张量的元素是否有界。
 * 计算图：如下
 * 场景一：非浮点数，肯定有界，直接返回True
 *
 * ```mermaid
 * graph LR
 *     A[(self)] --> C([l0op::Fill]) --> H([l0op::ViewCopy]) --> K[(out)]
 * ```
 *
 * 场景二：浮点数的情况，直接调IsFinite
 *
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous]) --> C([l0op::IsFinite]) --> H([l0op::ViewCopy]) --> K[(out)]
 * ```
 */

/**
 * @brief aclnnIsFinite的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: 原始张量。npu device侧的aclTensor，
 * 数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，支持非连续的Tensor。数据格式支持ND。
 * @param [out] out: npu device侧的aclTensor，数据类型只能是BOOL，shape与self一致，支持非连续的Tensor。数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnIsFiniteGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIsFinite的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnIsFiniteGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnIsFinite(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ISFINITE_H_
