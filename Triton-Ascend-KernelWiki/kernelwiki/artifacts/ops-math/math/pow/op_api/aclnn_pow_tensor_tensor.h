/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_POW_TENSOR_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_POW_TENSOR_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnPowTensorTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、
 * INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128，且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与other一致。
 * @param [in] exponent: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、
 * INT64、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128，且数据类型需要与self构成互相推导关系，shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in] out: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16、DOUBLE、INT32、INT64、
 * INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128，且数据类型需要是self与other推导之后可转换的数据类型，
 * shape为self与other两者broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent,
                                                           aclTensor* out, uint64_t* workspaceSize,
                                                           aclOpExecutor** executor);

/**
 * @brief aclnnInplacePowTensorTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplacePowTensorTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* exponent,
                                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnPowTensorTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnPowTensorTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnPowTensorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplacePowTensorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_POW_TENSOR_TENSOR_H_
