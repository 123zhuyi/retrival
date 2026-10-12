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
 * \file aclnn_addcdiv.h
 * \brief
 */

#ifndef OP_API_INC_ADDCDIV_H_
#define OP_API_INC_ADDCDIV_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddcdiv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：执行 tensor1 除以 tensor2 的元素除法，将结果乘以标量 value 并将其添加到 input
 * 计算公式：
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] self: npu
 * 需要累加的张量，npu device侧的aclTensor，数据类型支持DT_BFLOAT16、DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 *
 数据类型要和tensor1、tensor2支持可推导关系，shape要与tensor1、tensor2除过之后的tensor满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] tensor1: npu
 * 分子张量，npu device侧的aclTensor，数据类型支持DT_BFLOAT16、DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 * 数据类型要和self、tensor2支持可推导关系，shape要与tensor2满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] tensor2: npu
 * 分母张量，npu device侧的aclTensor，数据类型支持DT_BFLOAT16、DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 * 数据类型要和self、tensor1支持可推导关系，shape要与tensor1满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] value: host侧的aclScalar，数据类型与self、tensor1、tensor2数据类型保持一致，不一致时需要强转成一致。
 * @param [in] out: npu
 * 输出张量，npu device侧的aclTensor，数据类型支持DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 * 数据类型要满足和self、tensor1、tensor2推导关系，shape是self、tensor1、tensor2
 broadcast之后的shape，支持非连续Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddcdivGetWorkspaceSize(
    const aclTensor* self, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnAddcdiv的第二段接口，用于执行计算。
 *
 * * 算子功能：执行 tensor1 除以 tensor2 的元素除法，将结果乘以标量 value 并将其添加到 input
 * 计算公式：
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddcdivGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnAddcdiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnInplaceAddcdiv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：执行 tensor1 除以 tensor2 的元素除法，将结果乘以标量 value 并将其添加到 input
 * 计算公式：
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] selfRef: npu
 * 需要累加的张量，npu device侧的aclTensor，数据类型支持DT_BFLOAT16、DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 *
 数据类型要和tensor1、tensor2支持可推导关系，shape要与tensor1、tensor2除过之后的tensor满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] tensor1: npu
 * 分子张量，npu device侧的aclTensor，数据类型支持DT_BFLOAT16、DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 * 数据类型要和self、tensor2支持可推导关系，shape要与tensor2满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] tensor2: npu
 * 分母张量，npu device侧的aclTensor，数据类型支持DT_BFLOAT16、DT_FLOAT16、DT_FLOAT、DT_DOUBLE、DT_INT64，
 * 数据类型要和self、tensor1支持可推导关系，shape要与tensor1满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] value: host侧的aclScalar，数据类型与self、tensor1、tensor2数据类型保持一致，不一致时需要强转成一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* tensor1, const aclTensor* tensor2, const aclScalar* value,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceAddcdiv的第二段接口，用于执行计算。
 *
 * * 算子功能：执行 tensor1 除以 tensor2 的元素除法，将结果乘以标量 value 并将其添加到 input
 * 计算公式：
 * $$ out_i = self_i + value \times {tensor1_i \over tensor2_i} $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 graph LR
 *  A[(self)] -->B([l0op::Contiguous])
 *  B --> K([l0op::Cast])
 *  K --> C([l0op::Addcdiv])
 *  D[(tensor1)] -->E([l0op::Contiguous])
 *  E --> L([l0op::Cast])
 *  L --> C
 *  F[(tensor2)] --> G([l0op::Contiguous])
 *  G --> M([l0op::Cast])
 *  M --> C
 *  H((value)) --> C
 *  C --> O([l0op::Cast])
 *  O --> I([l0op::ViewCopy])
 *  I --> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddcdivGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceAddcdiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ADDCDIV_H_
