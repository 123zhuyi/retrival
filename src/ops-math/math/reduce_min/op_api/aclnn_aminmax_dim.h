/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_AMINMAX_DIM_H_
#define OP_API_INC_AMINMAX_DIM_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAminmaxDim的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：算子功能：计算输入张量的最小值和最大值。
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持整型，浮点类型。支持非连续的Tensor，数据格式支持ND。
 * @param [in] dim: host侧的Int64_t，指定要缩减的维度。
 * @param [in] keepDim: host侧的bool，reduce轴的维度是否保留。
 * @param [in] minOut: npu device侧的aclTensor，数据类型支持整型，浮点类型。支持非连续的Tensor，数据格式支持ND。
 * @param [in] maxOut: npu device侧的aclTensor，数据类型支持整型，浮点类型。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor* self, const int64_t dim, bool keepDim,
                                                      aclTensor* minOut, aclTensor* maxOut, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor);

/**
 * @brief aclnnAminmaxDim的第二段接口，用于执行计算。
 *
 * 算子功能：计算输入张量的最小值和最大值。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAminmaxDimGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAminmaxDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_AMINMAX_DIM_H_
