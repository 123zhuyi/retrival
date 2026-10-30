/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_TRIANGULAR_SOLVE_H_
#define OP_API_INC_TRIANGULAR_SOLVE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnTriangularSolve的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor求解AX=b
 * @param [in] self: 公式中的b, 数据类型支持FLOAT、DOUBLE、COMPLEX64，COMPLEX128，数据维度至少为2，且不大于8，
 * 支持非连续的Tensor，数据格式支持ND，且shape(*,m,n)需要与A(*,m,m)除最后两维外满足broadcast关系。
 * @param [in] A: 公式中的A, 数据类型支持FLOAT、DOUBLE、COMPLEX64，COMPLEX128，数据维度至少为2，且不大于8，
 * 支持非连续的Tensor，数据格式支持ND，且shape(*,m,n)需要与self(*,m,n)除最后两维外满足broadcast关系。
 * @param [in] upper: 计算属性，默认为true，A为上三角方阵，当upper为false时，A为下三角方阵
 * @param [in] transpose: 计算属性，默认为false，当transpose为true时，计算ATX=B,AT为A的转置
 * @param [in] unitriangular: 计算属性，默认为false，当unitriangular为true时，A的对角线元素视为1，而不是从A引用
 * @param [out] xOut: 公式中的X，数据类型支持FLOAT、DOUBLE、COMPLEX64，COMPLEX128，数据维度至少为2，且不大于8，
 * 支持非连续的Tensor，数据格式支持ND，且shape需要与broadcast后的A,b满足AX=b约束
 * @param [out] mOut: broadcast后A的上三角（下三角）拷贝，数据类型支持FLOAT、DOUBLE、COMPLEX64，COMPLEX128，
 * 数据维度至少为2，且不大于8，支持非连续的Tensor，数据格式支持ND，且shape需要与broadcast后的A一致
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnTriangularSolveGetWorkspaceSize(const aclTensor* self, const aclTensor* A, bool upper,
                                                           bool transpose, bool unitriangular, aclTensor* xOut,
                                                           aclTensor* mOut, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief: aclnnTriangularSolve的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成求解操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnTriangularSolveGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnTriangularSolve(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_TRIANGULAR_SOLVE_H_