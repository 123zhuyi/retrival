/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_BITWISE_AND_H_
#define OP_API_INC_BITWISE_AND_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBitwiseAnd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([ViewCopy])
 *     H --> I([Cast])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与other构成互相推导关系，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与self构成互相推导关系，
 * shape需要与self满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要是self与other推导之后可转换的数据类型，
 * shape需要是self与other，broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseAndTensorOutGetWorkspaceSize(const aclTensor* self, const aclTensor* other,
                                                               aclTensor* out, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);

/**
 * @brief aclnnBitwiseAnd的第二段接口，用于执行计算。
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([ViewCopy])
 *     H --> I([Cast])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnBitwiseAndTensorOutGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBitwiseAndTensorOut(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

/**
 * @brief aclnnInplaceBitwiseAnd的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([ViewCopy])
 *     H --> I([Cast])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与other构成互相推导关系，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持INT16,UINT16,INT32,INT64,INT8,UINT8,BOOL，且数据类型需要与self构成互相推导关系，
 * shape需要与self满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize(const aclTensor* self, const aclTensor* other,
                                                                      uint64_t* workspaceSize,
                                                                      aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBitwiseAnd的第二段接口，用于执行计算。
 *
 * 算子功能：完成位与或者逻辑与计算
 * 计算公式：
 * $$ output_i = self_i\&other_i $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([Contiguous])
 *     B --> C([Cast])
 *     C --> G([BitwiseAnd])
 *     D[(other)] -->E([Contiguous])
 *     E --> F([Cast])
 *     F --> G
 *     G --> H([ViewCopy])
 *     H --> I([Cast])
 *     I --> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBitwiseAndTensorOut(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_BITWISE_AND_H_
