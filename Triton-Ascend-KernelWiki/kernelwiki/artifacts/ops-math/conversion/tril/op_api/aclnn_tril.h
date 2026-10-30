/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_TRIL_H_
#define OP_API_INC_LEVEL2_ACLNN_TRIL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTril的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：返回输入矩阵（2-D `Tensor`）或一批矩阵的下三角部分，结果`Tensor`out的其他元素设置为0。

 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0::Contiguous])
 *   B --> E([l0::Tril])
 *   C[(diagonal)] --> E
 *   E --> G([l0::ViewCopy])
 *   G --> H[(out)]
 * ```
 *
 * @param [in] self: 待进行tril计算的入参。npu device侧的aclTensor，
 * 数据类型支持DOUBLE,FLOAT,FLOAT16,INT16,INT32,INT64,INT8,UINT16,UINT32,UINT64,UINT8,BOOL。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] diagonal: 对角线的位置，数据类型支持INT。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTrilGetWorkspaceSize(
    const aclTensor* self, int64_t diagonal, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnTril的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnTril(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnInplaceTril的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：返回输入矩阵（2-D `Tensor`）或一批矩阵的下三角部分，结果`Tensor`out的其他元素设置为0。
 *
 * @param [in] selfRef: 待进行tril计算的入参。npu device侧的aclTensor，
 * 数据类型支持DOUBLE,FLOAT,FLOAT16,INT16,INT32,INT64,INT8,UINT16,UINT32,UINT64,UINT8,BOOL。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] diagonal: 对角线的位置，数据类型支持INT。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceTrilGetWorkspaceSize(
    const aclTensor* selfRef, int64_t diagonal, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceTril的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAllGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceTril(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_TRIL_H_