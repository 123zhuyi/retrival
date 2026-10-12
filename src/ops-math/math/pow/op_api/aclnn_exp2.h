/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_EXP2_H_
#define OP_API_INC_LEVEL2_ACLNN_EXP2_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。
 * 计算公式：如下
 * $$
 *     out_{i} = 2^{self_{i}}
 * $$
 * @brief aclnnExp2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品，昇腾950 AI处理器：
 * 数据类型支持BOOL，INT8，UINT8，INT16，INT32，INT64，FLOAT，FLOAT16，BFLOAT16，DOUBLE。
 * Atlas 训练系列产品，Atlas 200I/500 A2 推理产品：数据类型支持BOOL，INT8，UINT8，INT16，INT32，INT64，FLOAT，FLOAT16，DOUBLE。
 * @param [in] out: npu device侧的aclTensor，数据类型需要是self可转换的数据类型，shape需要与self一致，支持非连续的Tensor，数据
 * 格式支持ND，数据维度不支持8维以上。
 * Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品，昇腾950 AI处理器：
 * 数据类型支持FLOAT，FLOAT16，BFLOAT16，DOUBLE。
 * Atlas 训练系列产品，Atlas 200I/500 A2 推理产品：数据类型支持FLOAT，FLOAT16，DOUBLE。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExp2GetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief aclnnExp2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnExp2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExp2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceExp2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] selfRef: npu device侧的aclTensor，支持非连续的Tensor，数据格式支持ND。
 * Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品，昇腾950 AI处理器：
 * 数据类型支持FLOAT，FLOAT16，BFLOAT16，DOUBLE。
 * Atlas 训练系列产品，Atlas 200I/500 A2 推理产品：数据类型支持FLOAT，FLOAT16，DOUBLE。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceExp2GetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnInplaceExp2的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceExp2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceExp2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_EXP2_H_
