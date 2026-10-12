/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SQUARE_H_
#define OP_API_INC_LEVEL2_ACLNN_SQUARE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSquare的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：返回输入Tensor中每个元素平方的结果
 * 计算公式：
 * $$ out_{i} = self_{i} \times self_{i} $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)]--->B([l0op::Contiguous])
 *   B--->C([l0op::Square])
 *   C--->D([l0op::ViewCopy])
 *   D--->E[(out)]
 * ```
 *
 * @param [in] self: 待进行square计算的入参。npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、INT64、INT32、BF16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] out: square计算的出参。npu device侧的aclTensor,
 * 数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、BF16，数据格式支持ND，支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包括算子计算流程
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnSquareGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSquare的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnSquareGetWorkspaceSize获取。
 * @param [in] exector: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus
aclnnSquare(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_SQUARE_H_