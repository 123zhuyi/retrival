/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_GESCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_GESCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGeScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：判断输入Tensor中的每个元素是否大于等于other Scalar的值，返回一个Bool类型的Tensor，
 * 对应输入Tensor中每个位置的大于等于判断是否成立
 * 计算公式： $$ out_{i}= (self_i >= other) ? True : False $$
 *
 * @param [in] self: 待进行ge计算的入参,npu device侧的aclTensor，
 * 数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BFLOAT16，数据格式支持ND,
 * 支持非连续的Tensor。
 * @param [in] other: 待进行ge计算的入参,aclScalar。
 * @param [in] out: ge计算的出参。npu device侧的aclTensor，
 * 输出一个数据类型为BOOL的Tensor，数据格式支持ND，
 * 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* other, aclTensor* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGeScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnGeScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

/**
 * @brief aclnnInplaceGeScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：判断输入Tensor中的每个元素是否大于等于other Scalar的值
 * 计算公式： $$ selfRef_{i} = (selfRef_{i} >= other_{i}) ? True : False $$
 *
 * @param [in] selfRef: 待进行ge本地计算的入参,npu device侧的aclTensor，
 * 数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: 待进行ge计算的入参,aclScalar。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceGeScalarGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceGeScalar的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceGeScalarGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceGeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_GESCALAR_H_
