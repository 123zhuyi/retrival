/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMMAX_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMMAX_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：计算self中的累积最大值，并返回该值以及其对应的索引
 * 计算公式：
 * $self_{i}$是输入张量self中，从维度dim视角来看的某个元素（其它维度下标不变，只dim维度下标依次递增），
 * $$
 * valuesOut_{i} = max(self_{1}, self_{2}, self_{3}, ......, self_{i})
 * $$
 * $$
 * indicesOut_{i} = argmax(self_{1}, self_{2}, self_{3}, ......, self_{i})
 * $$
 */

/**
 * @brief aclnnCummax的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、
 * FLOAT16、BFLOAT16、BOOL，数据类型需要能转换成out的数据类型，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] dim: host侧的整数，数据类型支持INT64。
 * @param [in] valuesOut：npu device侧的aclTensor，数据类型支持FLOAT、DOUBLE、UINT8、INT8、INT16、INT32、INT64、
 * FLOAT16、BFLOAT16、BOOL，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上, 且shape必须与self一致。
 * @param [in] indicesOut：npu device侧的aclTensor，数据类型支持INT32、INT64。支持非连续的Tensor，数据格式支持ND，
 * 数据维度不支持8维以上, 且shape必须与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCummaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* valuesOut,
                                                  aclTensor* indicesOut, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor);

/**
 * @brief aclnnCummax的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCummaxGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCummax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CUMMAX_H_
