/**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_LE_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_LE_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLeTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：计算两个Tensor中的元素是否小于等于other的值，将self每个元素与other的值的比较结果写入out中
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持INT8,UINT8,INT16,UINT16,INT32,INT64,FLOAT16,FLOAT,DOUBLE数据类型，
 * 数据类型需要与other构成相互推导关系，shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持INT8,UINT8,INT16,UINT16,INT32,INT64,FLOAT16,FLOAT,DOUBLE数据类型，
 * 数据类型需要与other构成相互推导关系，shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND
 * @param [in] out: npu device侧的aclTensor，
 * 数据类型支持INT8,UINT8,INT16,UINT16,INT32,INT64,FLOAT16,FLOAT,DOUBLE数据类型，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLeTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLeTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLeTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLeTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceLeTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：判断输入Tensor中的每个元素是否小于等于other Tensor的值
 * 计算公式： $$ selfRef_{i} = (selfRef_{i} <= other_{i}) ? True : False $$
 *
 * @param [in] selfRef: 待进行le本地计算的入参,npu device侧的aclTensor，
 * 数据类型支持FLOAT16、FLOAT32、INT32、INT64、INT8、UINT8、DOUBLE、UINT16、UINT32、UINT64、BOOL、BFLOAT16、BFLOAT16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: 待进行ge计算的入参,aclTensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLeTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLeTensor的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceLeTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceLeTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_LE_TENSOR_H_
