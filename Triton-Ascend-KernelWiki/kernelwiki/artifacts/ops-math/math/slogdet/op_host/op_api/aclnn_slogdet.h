/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_API_INC_SLOGDET_H_
#define OP_API_INC_SLOGDET_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSlogdet的第一段接口，根据具体的计算流程，计算workspace大小.
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor计算行列式的符号和自然对数
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(self)]-->B([Contiguous])-->C([LogMatrixDeterminant])
 * D1([Cast])-->E1([ViewCopy])-->F1([signOut])
 * D2([Cast])-->E2([ViewCopy])-->F2([logOut])
 * ```
 *
 * @param [in] self: npu device侧的aclTensor, 数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128. shape满足(*, n, n)形式,
 * 其中`*`表示0或更多维度的batch. 支持非连续的Tensor. 数据格式支持ND.
 * @param [in] signOut: npu device侧的aclTensor, 数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128.
 * shape和self的batch一致. 支持非连续的Tensor. 数据格式支持ND.
 * @param [in] logOut: npu device侧的aclTensor, 数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128.
 * shape和self的batch一致. 支持非连续的Tensor. 数据格式支持ND.
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小.
 * @param [out] executor: 返回op执行器，包含算子计算流程.
 * @return aclnnStatus: 返回状态码， 成功返回ACLNN_SUCCESS, 失败返回对应错误码.
 */
ACLNN_API aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor* self, aclTensor* signOut, aclTensor* logOut,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSlogdet的第二段接口，用于执行计算.
 *
 * 算子功能：对输入的每个元素完成相反数计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址.
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSlogdetGetWorkspaceSize获取.
 * @param [in] executor: op执行器，包含了算子计算流程.
 * @param [in] stream: acl stream流.
 * @return aclnnStatus: 返回状态码.
 */
ACLNN_API aclnnStatus aclnnSlogdet(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SLOGDET_H_
