/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_addcmul.h
 * \brief
 */

#ifndef OP_API_INC_ADD_CMUL_H_
#define OP_API_INC_ADD_CMUL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddcmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成乘加计算
 * 计算公式：
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与其他输入构成互相推导关系，shape需要与其他输入满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与其他输入一致。
 * @param [in] tensor1: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与其他输入构成互相推导关系，shape需要与其他输入满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与其他输入一致。
 * @param [in] tensor2: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与其他输入构成互相推导关系，shape需要与其他输入满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与其他输入一致。
 * @param [in] value: host侧的aclScalar，数据类型需要可转换成其他输入推导后的数据类型。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要是其他输入推导之后可转换的数据类型，shape需要是其他输入
 * broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddcmulGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddcmul的第二段接口，用于执行计算。
 *
 * 算子功能：完成乘加计算
 * 计算公式：
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnAddcmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceAddcmul的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成乘加计算
 * 计算公式：
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与其他输入构成互相推导关系，shape需要与其他输入满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与其他输入一致。
 * @param [in] tensor1: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与其他输入构成互相推导关系，shape需要与其他输入满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与其他输入一致。
 * @param [in] tensor2: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与其他输入构成互相推导关系，shape需要与其他输入满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与其他输入一致。
 * @param [in] value: host侧的aclScalar，数据类型需要可转换成其他输入推导后的数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceAddcmul的第二段接口，用于执行计算。
 *
 * 算子功能：完成乘加计算
 * 计算公式：
 * $$ output=self+ value \times tensor1 \times tensor2 $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]--> B([l0op::Contiguous]) --> C([l0op::Cast]) -->D([Addcmul])
 * E[(tensor1)]--> B1([l0op::Contiguous]) --> G([l0op::Cast]) --> D
 * E1[(tensor2)]--> B2([l0op::Contiguous]) --> G1([l0op::Cast])  --> D
 * D --> H([l0op::Cast]) --> I([l0op::ViewCopy]) --> J[(out)]
 * K((value)) --> L([l0op::Cast]) --> D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceAddcmul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ADD_CMUL_H_
