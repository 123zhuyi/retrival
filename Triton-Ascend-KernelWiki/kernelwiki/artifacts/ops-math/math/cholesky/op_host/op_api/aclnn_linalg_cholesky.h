/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OP_API_INC_LINALG_CHOLESKY_H_
#define OP_API_INC_LINALG_CHOLESKY_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ACLNN_MAX_SHAPE_RANK 8

/**
 * @brief aclnnLinalgCholesky的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算实数对称正定矩阵的Cholesky分解。
 * 计算公式：
 * $$ self = out \times out^{H} $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0op::Contiguous])
 *     B --> C([l0op::Transpose])
 *     C --> D([l0op::Cast])
 *     D --> E([l0op::Cholesky])
 *     E --> F([l0op::Cast])
 *     F --> G([l0op::Transpose])
 *     G --> H([l0op::ViewCopy])
 *     H --> I[(out)]
 *     J[(upper)] --> E
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16。支持非连续的Tensor，数据格式支持ND。
 * @param [in] upper: npu host侧的布尔型，True表示分解为上三角矩阵，False表示分解为下三角矩阵。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16。数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor* self, bool upper, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnLinalgCholesky的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLinalgCholeskyGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLinalgCholesky(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LINALG_CHOLESKY_H_