/**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_LE_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_LE_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLeScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 计算self中的元素的值是否小于等于other的值，将self每个元素与other的值的比较结果写入out中
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持INT8,UINT8,INT16,UINT16,INT32,INT64,FLOAT16,FLOAT,DOUBLE。数据格式支持ND，支持非连续的Tensor。
 * @param [in] other: host侧的aclScalar，数据类型支持INT8,UINT8,INT16,UINT16,INT32,INT64,FLOAT16,FLOAT,DOUBLE
 * 数据类型需要能转换成self的数据类型
 * @param [in] out: npu device侧的aclTensor，数据类型支持INT8,UINT8,INT16,UINT16,INT32,INT64,FLOAT16,FLOAT,DOUBLE
 * shape与self的shape一致，数据格式支持ND
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLeScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLeScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLeScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceLeScalarGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：计算selfRef Tensor中的元素是否小于等于other
 * Scalar中的元素，返回修改后的selfRef，selfRef<=other的为True，否则为False
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,INT32,UINT8,BOOL,UINT64,UINT32,DOUBLE,UINT16，BFLOAT16数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: host侧的aclScalar，shape需要与other满足broadcast关系。
 * @param [in] out: npu device侧的aclTensor，输出一个数据类型为BOOL类型的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLeScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLeScalar的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLeScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceLeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_LE_SCALAR_H_
