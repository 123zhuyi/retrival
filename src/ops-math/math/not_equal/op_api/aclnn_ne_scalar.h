/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_NE_SCALAR_H_
#define OP_API_INC_LEVEL2_ACLNN_NE_SCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNeScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,UINT64,INT32,INT8,UINT8,BOOL,UINT32,BFLOAT16,INT16，数据格式支持ND，支持非连续的Tensor。
 * @param [in] other:
 * host侧的aclScalar，数据类型支持FLOAT16,FLOAT,INT64,UINT64,INT32,INT8,UINT8,BOOL,UINT32,BFLOAT16,INT16,
 * 数据类型需要能转换成self的数据类型
 * @param [in] out: npu device侧的aclTensor，数据类型为bool，shape与self的shape一致，数据格式支持ND
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNeScalarGetWorkspaceSize(const aclTensor* self, const aclScalar* other, aclTensor* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnNeScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnNeScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    const aclrtStream stream);

/**
 * @brief aclnnInplaceNeScalar的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,UINT64,INT32,INT8,UINT8,BOOL,UINT32,BFLOAT16,INT16数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。selfRef也是输出结果。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,UINT64,INT32,INT8,UINT8,BOOL,UINT32,BFLOAT16,INT16数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND
 * @param [in] out: npu device侧的aclTensor，输出一个数据类型为BOOL类型的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceNeScalarGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceNeScalar的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceNeScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceNeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_NE_SCALAR_H_