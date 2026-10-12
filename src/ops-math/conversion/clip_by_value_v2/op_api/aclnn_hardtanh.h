/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_HARDTANH_H_
#define OP_API_INC_LEVEL2_ACLNN_HARDTANH_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnHardtanh的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于max则限制为max，
 * 若元素小于min则限制为min，否则等于元素本身，min默认值为-1.0，max默认值为1.0。
 * 计算公式：如下所示
 * $$
 * HardTanh(x) = \left\{\begin{matrix}
 * \begin{array}{l}
 * clipValueMax\ \ \ \ \ \ \ if \ \ x>clipValueMax \\
 * clipValueMin\ \ \ \ \ \ \ if\ \ x<clipValueMin \\
 * x\ \ \ \ \ \ \ \ \ \ \ otherwise \\
 * \end{array}
 * \end{matrix}\right.\begin{array}{l}
 * \end{array}
 * $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0::Contiguous])
 *     B --> E([l0::ClipByValue])
 *     C((clipValueMin)) --> E
 *     D((clipValueMax)) --> E
 *     E --> G([l0::ViewCopy])
 *     G --> H[(out)]
 * ```
 *
 * @param [in] self: 待进行erf计算的入参。npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、
 * UINT8、FLOAT64，数据格式支持ND，且数据格式需要与out一致，支持非连续的Tensor。
 * @param [in] out: erf计算的出参。npu
 * device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、
 * FLOAT64，数据格式支持ND，且数据格式需要与self一致， 支持非连续的Tensor。
 * @param [in] clipValueMin: host侧的aclScalar，数据类型需要可转换成self的数据类型。
 * @param [in] clipValueMax: host侧的aclScalar，数据类型需要可转换成self的数据类型。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnHardtanhGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnHardtanh的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnHardtanhGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnHardtanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceHardtanh的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于max则限制为max，
 * 若元素小于min则限制为min，否则等于元素本身，min默认值为-1.0，max默认值为1.0。
 * 计算公式：如下所示
 * $$
 * HardTanh(x) = \left\{\begin{matrix}
 * \begin{array}{l}
 * clipValueMax\ \ \ \ \ \ \ if \ \ x>clipValueMax \\
 * clipValueMin\ \ \ \ \ \ \ if\ \ x<clipValueMin \\
 * x\ \ \ \ \ \ \ \ \ \ \ otherwise \\
 * \end{array}
 * \end{matrix}\right.\begin{array}{l}
 * \end{array}
 * $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(self)] -->B([l0::Contiguous])
 *     B --> E([l0::ClipByValue])
 *     C((clipValueMin)) --> E
 *     D((clipValueMax)) --> E
 *     E --> G([l0::ViewCopy])
 *     G --> H[(out)]
 * ```
 *
 * @param [in] selfRef: 待进行erf计算的入参。npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、
 * UINT8、FLOAT64，数据格式支持ND，且数据格式需要与out一致， 支持非连续的Tensor。
 * @param [in] clipValueMin: host侧的aclScalar，数据类型需要可转换成self的数据类型。
 * @param [in] clipValueMax: host侧的aclScalar，数据类型需要可转换成self的数据类型。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceHardtanhGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* clipValueMin, const aclScalar* clipValueMax, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceHardtanh的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceHardtanhGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceHardtanh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_HARDTANH_H_