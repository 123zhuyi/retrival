/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMSUM_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMSUM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中
 * 计算公式：
 * $x_{i}$是输入张量self中，从维度dim视角来看的某个元素（其它维度下标不变，只dim维度下标依次递增），
 * $y_{i}$是输出张量out中对应位置的元素，则
 *
 * $$
 * y_{i} = x_{1} + x_{2} + x_{3} + ...... + x_{i}
 * $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(self)] --> B([l0op::Contiguous])
 *     B --> C([l0op::Cast])
 *     C --> D([l0op::Cumsum])
 *     E((dim)) --> D
 *     F((dtype)) --> C
 *     D --> G([l0op::ViewCopy])
 *     G --> H[(out)]
 * ```
 */

/**
 * @brief aclnnCumsum的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128、UINT8、INT8、INT16、INT32、
 * INT64、FLOAT16、BFLOAT16、BOOL，数据类型需要能转换成out的数据类型，支持非连续的Tensor，数据格式支持ND，数据维度不支持8维以上。
 * @param [in] dim: host侧的整数，数据类型支持INT64。
 * @param [in]
 * dtype：host侧的数据类型枚举，表示输出张量所需的数据类型，支持FLOAT、FLOAT16、INT32、DOUBLE、UINT8、INT8、INT16、
 * INT64、COMPLEX64、COMPLEX128、BFLOAT16（910B支持），且需与输出张量out的数据类型一致。
 * @param [in] out：npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32、DOUBLE、UINT8、INT8、INT16、INT64、
 * COMPLEX64、COMPLEX128、BFLOAT16（910B支持），且数据类型需要与传入的dtype一致，shape需要与self一致，支持非连续的Tensor，数据
 * 格式支持ND，数据维度不支持8维以上。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor* self, int64_t dim, aclDataType dtype, aclTensor* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnCumsum的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCumsumGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCumsum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnCumsumV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnCumsumV2GetWorkspaceSize(const aclTensor* self, int64_t dim, bool exclusive, bool reverse,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnCumsumV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_CUMSUM_H_
