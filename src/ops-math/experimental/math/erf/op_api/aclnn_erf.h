/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ERF_H_
#define OP_API_INC_LEVEL2_ACLNN_ERF_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnErf的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素对应的误差函数的值
 * 计算公式：
 * $$ erf(x)=\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * 计算图一：如下所示
 * 场景：当输入类型在Erf算子支持的范围之内（FLOAT32、FLOAT16、BFLOAT16、FLOAT64）时，使用Erf算子完成计算。
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B -->C([l0op::Erf])
 *     C --> D([l0op::ViewCopy])
 *     D --> E[(out)]
 * ```
 *
 * 计算图二：如下所示
 * 场景：self的数据类型为BOOL，将self的数据类型CAST为FLOAT32，再使用Erf算子完成计算。
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B([l0op::Contiguous])
 *     B -->C([l0op::Cast])
 *     C -->D([l0op::Erf])
 *     D -->H([l0op::Cast])
 *     H --> E([l0op::ViewCopy])
 *     E --> F[(out)]
 * ```
 *
 * @param [in] self: 待进行erf计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、FLOAT16、BFLOAT16、BOOL、INT64，数据格式支持ND， 支持非连续的Tensor。
 * @param [in] out: erf计算的出参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、FLOAT16、BFLOAT16，数据格式支持ND， 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnErfGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnErf的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnErfGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnErf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceErf的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * 算子功能：返回输入Tensor中每个元素对应的误差函数的值
 * 计算公式：
 * $$ erf(x)=\frac{2}{\sqrt{\pi } } \int_{0}^{x} e^{-t^{2} } \mathrm{d}t $$
 *
 * 计算图：如下所示
 * ```mermaid
 * graph LR
 *     A[(Self)]  --> B{l0op::Contiguous}
 *     B -->C([l0op::Erf])
 *     C --> D{l0op::ViewCopy}
 *     D --> E[(out)]
 * ```
 *
 * @param [in] selfRef: 待进行erf计算的入参。npu device侧的aclTensor，
 * 数据类型支持FLOAT64、FLOAT32、FLOAT16、BFLOAT16、BOOL，数据格式支持ND， 支持非连续的Tensor。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceErfGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceErf的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnErfGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceErf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_ERF_H_