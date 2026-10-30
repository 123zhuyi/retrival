/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_logdet.h
 * @brief ACLNN L2 API - Logdet 算子接口声明
 *
 * 算子功能：计算输入方阵（或方阵 batch）行列式的自然对数。
 * 采用两段式设计：
 * - aclnnLogdetGetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnnLogdet: 执行计算
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_LOGDET_H_
#define OP_API_INC_LEVEL2_ACLNN_LOGDET_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLogdet 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_math
 *
 * 算子功能：计算输入方阵行列式的自然对数 log(|det(A)|)。
 * 对于奇异矩阵（行列式为 0），返回 -inf。
 * 对于负行列式矩阵，返回 NaN。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0::Contiguous])
 *   B --> E([l0::Logdet])
 *   E --> G([l0::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * @param [in] self: npu device 侧的 aclTensor，数据类型支持 FLOAT。
 *        shape 满足 (*, n, n) 形式，其中 * 表示 0 或更多维度的 batch。
 *        支持非连续的 Tensor，数据格式支持 ND。
 * @param [in] out: npu device 侧的 aclTensor，数据类型支持 FLOAT。
 *        shape 和 self 的 batch 一致（去掉最后两维）。
 *        支持非连续的 Tensor，数据格式支持 ND。
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLogdetGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLogdet 的第二段接口，用于执行计算。
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，
 *        由第一段接口 aclnnLogdetGetWorkspaceSize 获取。
 * @param [in] executor: op 执行器，包含了算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLogdet(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_LOGDET_H_
