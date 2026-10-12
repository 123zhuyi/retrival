/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 *
 算子功能：计算输入张量self中每个元素和输入标量other的按位或。输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑或。
 * 计算公式：如下
 * $$
 * \text{out}_i =
 * \text{self}_i \, | \, \text{other}
 * $$
 *
 * 实现说明：如下
 * 计算图一：如下
 * 场景：经过类型推导后，self和other的数据类型都为BOOL类型时，需要调用l0::LogicalOr接口做计算

 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous])
 * B --> C([l0op::Cast])
 * C --> D([l0op::LogicalOr])
 * E((other)) --> D
 * D --> F([l0op::Cast])
 * F --> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 *
 * 计算图二：如下
 * 场景：不满足计算图一的条件时，都会调用l0::BitwiseOr接口做计算
 *
 * ```mermaid
 * graph LR
 * A[(self)] --> B([l0op::Contiguous])
 * B --> C([l0op::Cast])
 * C --> D([l0op::BitwiseOr])
 * E((other)) --> D
 * D --> F([l0op::Cast])
 * F --> G([l0op::ViewCopy])
 * G --> H[(out)]
 * ```
 */

/**
 * @brief aclnnBitwiseOrScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，
 * 昇腾950 AI处理器：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64。
 * 其他系列产品：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、INT64。
 * 数据类型与other的数据类型需满足数据类型推导规则，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] other：host侧的aclScalar，
 * 昇腾950 AI处理器：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64。
 * 其他系列产品：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、INT64。
 * 数据类型与self的数据类型需满足数据类型推导规则。
 * @param [in] out: npu device侧的aclTensor，数据类型需要是self与other推导之后可转换的数据类型，shape需要与self一致，支持非连续的
 * Tensor，数据格式支持ND，数据维度不支持8维以上。
 * Atlas 训练系列产品，Atlas 推理系列产品：数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8、FLOAT、FLOAT16、DOUBLE。
 * Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：
 * 数据类型支持BOOL、INT8、INT16、UINT16、INT32、INT64、UINT8、FLOAT、FLOAT16、DOUBLE、BFLOAT16。
 * 昇腾950 AI处理器：数据类型支持BOOL、INT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、UINT8、FLOAT、FLOAT16、DOUBLE、
 * BFLOAT16、COMPLEX64、COMPLEX128。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseOrScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* other,
                                                           aclTensor* out, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief aclnnBitwiseOrScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnBitwiseOrScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseOrScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

/**
 * @brief aclnnInplaceBitwiseOrScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] selfRef: npu device侧的aclTensor，
 * 昇腾950 AI处理器：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64。
 * 其他系列产品：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、INT64。
 * 数据类型与other的数据类型需满足数据类型推导规则，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in]
 * other：host侧的aclScalar，
 * 昇腾950 AI处理器：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64。
 * 其他系列产品：数据类型支持BOOL、INT8、UINT8、INT16、UINT16、INT32、INT64。
 * 数据类型与self的数据类型需满足数据类型推导规则。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseOrScalarGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other,
                                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBitwiseOrScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceBitwiseOrScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseOrScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_BITWISE_OR_SCALAR_H_
