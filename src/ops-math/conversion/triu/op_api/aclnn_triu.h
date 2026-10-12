/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_TRIU_H_
#define OP_API_INC_LEVEL2_ACLNN_TRIU_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTriu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：将输入的self张量的最后二维（按shape从左向右数）沿主对角线的左下部分置零。参数diagonal可正可负，
 * 默认为零，正数表示主对角线向右上方向移，负数表示主对角线向左下方向移动。
 * 计算公式：下面用i表示遍历倒数第二维元素的序号（i是行索引），用j表示遍历最后一维元素的序号（j是列索引），
 * 用d表示diagonal，在(i, j)对应的二维坐标图中，i+d==j表示在对角线上，
 * $$
 * 对角线及其右上方，即i+d<=j,保留原值： out_{i,j} = self_{i,j}\\
 * 而位于对角线左下方的情况，即i+d>j,置零：out_{i,j} = 0
 * $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0::Contiguous]) --> C([l0op::Triu])
 *   C --> D([l0op::ViewCopy]) --> E[(out)]
 * ```
 *
 * @param [in] self: 待进行triu计算的入参。npu device侧的aclTensor，
 * 数据类型支持DOUBLE,FLOAT,FLOAT16,BFLOAT16(仅910B),INT16,INT32,INT64,INT8,UINT16,UINT32,UINT64,UINT8,BOOL。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] diagonal: 主对角线的偏移量，数据类型支持INT。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTriuGetWorkspaceSize(
    const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnTriu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTriu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceTriu的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持DOUBLE,FLOAT,FLOAT16,BFLOAT16(仅910B),INT16,INT32,INT64,INT8,UINT16,UINT32,UINT64,UINT8,BOOL。
 * 支持非连续的Tensor，数据格式支持ND
 * @param [in] diagonal: 主对角线的偏移量，数据类型支持INT。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceTriuGetWorkspaceSize(
    aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceTriu的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceTriuGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceTriu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_TRIU_H_
