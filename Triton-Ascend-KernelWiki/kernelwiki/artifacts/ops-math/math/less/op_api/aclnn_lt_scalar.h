/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LTSCALAR_H_
#define OP_API_INC_LTSCALAR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLtScalarGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算self Tensor中的元素是否小于(<)other
 * Scalar中的元素，返回一个Bool类型的Tensor，self<other的为True，否则为False 计算公式：$$ out = (self < other)  ? [True]
 * : [False] $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,INT32,UINT8,BOOL,UINT64,UINT32,DOUBLE,UINT16数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: host侧的aclScalar，shape需要与other满足broadcast关系。
 * @param [in] out: npu device侧的aclTensor，输出一个数据类型为BOOL类型的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnLtScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLtScalar的第二段接口，用于执行计算。
 *
 * 算子功能：计算self Tensor中的元素是否小于(<)other
 * Scalar中的元素，返回一个Bool类型的Tensor，self<other的为True，否则为False 计算公式：$$ out = (self < other)  ? [True]
 * : [False] $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLtScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLtScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceLtScalarGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算selfRef Tensor中的元素是否小于(<)other
 * Scalar中的元素，返回修改后的selfRef，selfRef<other的为True，否则为False 计算公式：$$ out = (selfRef < other)  ?
 * [True] : [False] $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(SelfRef)] --> B([l0op::Contiguous])
 * B-->C1([l0op::Cast])--> D([l0op::Less])
 * E((other)) --> C2([l0op::Cast])
 * C2 -->D
 * D --> C3([l0op::Cast])
 * C3 --> F1([l0op::ViewCopy])
 * F1 --> J[(selfRef)]
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT16,FLOAT,INT64,INT32,UINT8,BOOL,UINT64,UINT32,DOUBLE,UINT16数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: host侧的aclScalar，shape需要与other满足broadcast关系。
 * @param [in] out: npu device侧的aclTensor，输出一个数据类型为BOOL类型的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus aclnnInplaceLtScalarGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLtScalar的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLtScalarGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceLtScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LTSCALAR_H_
