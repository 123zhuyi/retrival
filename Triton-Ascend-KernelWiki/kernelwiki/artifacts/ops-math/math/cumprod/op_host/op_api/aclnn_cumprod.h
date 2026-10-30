/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCumprod的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能： 计算输入张量（Tensor）input沿着指定维度的累积乘积。它返回一个新的张量，其形状与输入张量相同
 * @param [in] input: device侧的aclTensor，表示要进行累积乘积运算的输入张量，数据类型支持DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16, DT_DOUBLE, 
    DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX128, DT_COMPLEX64
 * @param [in] dim: npu device侧的aclScalar，用于指定沿着哪个维度进行累积乘积运算,数据类型支持DT_INT32
 * @param [in] dtype host侧的aclDataType，输出tensor的数据类型，需要与out数据类型一致
 * @param [in] out: 返回一个新的张量，其形状与输入张量相同,包含累积乘积。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor *input, const aclScalar *dim, const aclDataType dtype,
                                                    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnCumprod的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能： 计算输入张量（Tensor）input沿着指定维度的累积乘积。它返回一个新的张量，其形状与输入张量相同
 * @param [in] input: device侧的aclTensor，表示要进行累积乘积运算的输入张量，数据类型支持DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16, DT_DOUBLE, 
    DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_COMPLEX128, DT_COMPLEX64
 * @param [in] dim: npu device侧的aclScalar，用于指定沿着哪个维度进行累积乘积运算,数据类型支持DT_INT32
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor *input, const aclScalar *dim, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief: aclnnCumprod的第二段接口，用于执行计算
 * @domain aclnn_ops_train
 * 算子功能： 计算输入的累积乘积
 * 该算子对应底层的tf算子
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCumprodGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

/**
 * @brief: aclnnCumprod的第二段接口，用于执行计算原地更新输入参数
 * @domain aclnn_ops_train
 * 算子功能： 计算输入的累积乘积
 * 该算子对应底层的tf算子
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceCumprodGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_