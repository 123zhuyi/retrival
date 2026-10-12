/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LOG_H_
#define OP_API_INC_LOG_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLog的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成自然对数的计算
 * 计算公式：
 * $$ output = log_e(self) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C -->D([Log])
 *     E[(Out)] -->F([Contiguous])
 *     F --> G([Cast])
 *     G -->D([Log])
 *     D --> H([Cast])
 *     H --> I([ViewCopy])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
 *                   支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLogGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                               aclOpExecutor** executor);

/**
 * @brief aclnnLog的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnLogGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceLog的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成自然对数的inplace计算。
 * 计算公式：
 * $$ selfRef_i = log_e(selfRef_i) $$
 *
 * @param [in] selfRef:
 * 公式中的输入`selfRef`，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLogGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLog的第二段接口，用于执行计算。
 *
 * 算子功能：完成自然对数的inplace计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLog(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LOG_H_
