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
 * \file aclnn_lerp_tensor.h
 * \brief
 */
#ifndef OP_API_ACLNN_LERP_TENSOR_H_
#define OP_API_ACLNN_LERP_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLerp的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。

 *
 * 计算图：
 * ```mermaid
 * graph LR
 * A1[(self)] --> B1([l0op::Contiguous])
 * A2[(end)] --> B2([l0op::Contiguous])
 * A3[(weight)] --> B3([l0op::Contiguous])
 * C([l0op::Lerp])
 * B1 --> C
 * B2 --> C
 * B3 --> C
 * C --> D([l0op::Cast])
 * D --> E([l0op::ViewCopy]) --> F[(out)]
 * ```
 *
 * @param [in] self: 公式中的输入start，数据类型支持FLOAT16、FLOAT，shape需要与end和weight满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] end: 公式中的输入end，数据类型支持FLOAT16、FLOAT且与self的数据类型一致，shape需要与self和weight满足broadcast关系。
 * 支持非连续Tensor，数据格式支持ND。
 * @param [in] weight: 公式中的输入weight，数据类型支持FLOAT16、FLOAT且与`self`的数据类型一致，shape需要与`self`和`end`满足broadcast关系。
 * 支持非连续Tensor，数据格式支持ND。
 * @param [in] out: 公式中的out，数据类型支持FLOAT16、FLOAT且与self的数据类型一致、shape与self、end和weight
 * broadcast之后的shape一致。支持非连续Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLerpGetWorkspaceSize(const aclTensor* self, const aclTensor* end, const aclTensor* weight,
                                                aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLerp的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnLerpGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLerp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

/**
 * @brief aclnnInplaceLerp的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。

 *
 * @param [in] selfRef: 公式中的输入start，数据类型支持FLOAT16、FLOAT，shape需要与end和weight满足broadcast关系，且broadcast后的shape与selfRef一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] end: 公式中的输入end，数据类型支持FLOAT16、FLOAT且与self的数据类型一致，shape需要与selfRef和weight满足broadcast关系，
 * 且broadcast后的shape与selfRef一致。 支持非连续Tensor，数据格式支持ND。
 * @param [in] weight: 公式中的输入weight，数据类型支持FLOAT16、FLOAT，shape需要与selfRef和end满足broadcast关系。支持非连续Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLerpGetWorkspaceSize(aclTensor* selfRef, const aclTensor* end,
                                                       const aclTensor* weight, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLerp的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceLerpGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLerp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_ACLNN_LERP_TENSOR_H_