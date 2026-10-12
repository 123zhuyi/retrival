/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_NEG_H_
#define OP_API_INC_NEG_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNeg的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对输入的每个元素完成相反数计算
 * 计算公式：
 * $$ output​=(-1) * self $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *      A[(self)] -->B([Contiguous])
 *      B -->D([Neg])
 *      D-->E([Cast])
 *      E-->F([ViewCopy])
 *      F-->G[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持整型，浮点类型，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor，数据类型支持整型，浮点类型，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnNegGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceNeg的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：对输入的每个元素完成相反数计算
 * 计算公式：
 * $$ selfRef ​=(-1) * selfRef $$
 *
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持整型，浮点类型，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceNegGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnNeg的第二段接口，用于执行计算。
 *
 * 算子功能：完成加法计算
 * 计算公式：
 * $$ output_i = self_i+alpha*other_i $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 *  graph LR
 *      A[(self)] -->B([Contiguous])
 *      B -->D([Neg])
 *      D-->E([Cast])
 *      E-->F([ViewCopy])
 *      F-->G[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnNegGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus
aclnnNeg(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnInplaceNeg的第二段接口，用于执行计算。
 *
 * 算子功能：对输入的每个元素完成相反数计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceNegGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceNeg(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif /* OP_API_INC_NEG_H_ */