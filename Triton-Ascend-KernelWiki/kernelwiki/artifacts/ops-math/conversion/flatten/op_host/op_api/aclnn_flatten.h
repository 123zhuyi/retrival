/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_FLATTEN_H_
#define OP_API_INC_FLATTEN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlatten的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：将输入Tensor，基于给定的axis，扁平化为一个2D的Tensor。
 * 若self的shape为(d_0,d_1,...,d_n)，那么输出out的shape为(d_0 X d_1 ... X d_(axis-1), d_axis X d_(axis+1)... X d_n)。
 * 若axis取值为0，则输出out的shape为(1, d_0 X d_1 ... X d_n)。
 * 实现说明-计算图
 *
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   B --> C([l0op::Flatten])
 *   D[(axis)] --> C
 *   C --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 *
 * @param [in]
 * self：输入`self`,数据类型支持INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16(Ascend910B)、FLOAT、FLOAT16。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] axis：输入`axis`，int64_t类型整数。表示flatten计算的基准轴。取值范围为[-self.dim(),self.dim())。
 * @param [out]
 * out：输出`out`,数据类型支持INT8、INT16、INT32、INT64、UINT8、BOOL、BFLOAT16(Ascend910B)、FLOAT、FLOAT16。
 * shape为2D。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor* self, int64_t axis, aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFlatten的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnFlattenGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFlatten(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_FLATTEN_H_
