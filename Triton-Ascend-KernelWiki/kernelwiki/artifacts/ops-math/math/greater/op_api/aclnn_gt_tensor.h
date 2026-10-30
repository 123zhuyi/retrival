/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GTTENSOR_H_
#define OP_API_INC_GTTENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGtTensor的第一段接口,根据具体的计算流程,计算workspace大小。
 * @domain aclnn_math
 * 计算self Tensor中的元素是否大于(>)other Tensor中的元素,返回一个Tensor,self>other的为True(1.),否则为False(0.):
 *
 * $$ out = (self_i > other_i)  ?  [True] : [False] $$
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A[(Self)] -->B([l0op::Contiguous])
 *     B -->C1([l0op::Cast])-->D([l0op::Greater])--> C3([l0op::Cast])
 *     E[(other)] -->F([l0op::Contiguous])
 *     F --> C2([l0op::Cast])-->D
 *     C3 -->F1([l0op::ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor,数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,UINT64,
 * UINT32,INT16,DOUBLE,UINT16,BFLOAT16数据类型,shape需要与other满足broadcast关系,
 * 数据类型需要与other满足数据类型推导规则,支持非连续的Tensor,数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor,数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,UINT64,
 * UINT32,INT16,DOUBLE,UINT16,BFLOAT16数据类型,shape需要与other满足broadcast关系,
 * 数据类型需要与self满足数据类型推导规则,支持非连续的Tensor,数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor,数据类型支持FLOAT16,FLOAT,INT64,INT32,INT8,UINT8,BOOL,UINT64,
 * UINT32,INT16,DOUBLE,UINT16,BFLOAT16数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器,包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGtTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGtTensor的第二段接口,用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小,由第一段接口aclnnGtTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器,包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGtTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceGtTensor的第一段接口,根据具体的计算流程,计算workspace大小。
 * @domain aclnn_math
 * 计算selfRef Tensor中的元素是否大于(>)other Tensor中的元素,返回结果复写到selfRef：
 *
 * $$ selfRef = (selfRef_i > other_i)  ?  [True] : [False] $$
 *
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *    A[(SelfRef)] -->B([l0op::Contiguous])
 *    B -->C1([l0op::Cast])-->D([l0op::Greater])--> C3([l0op::Cast])
 *    E[(other)] -->F([l0op::Contiguous])
 *    F --> C2([l0op::Cast])-->D
 *    C3 -->F1([l0op::ViewCopy])--> J[(SelfRef)]
 * ```
 *
 * @param [in] selfRef(aclTensor*): 数据类型支持UINT8,INT8,DOUBLE,FLOAT,INT32,INT64,INT16,FLOAT16,BOOL,BFLOAT16数据类型,
 * shape需要与other满足broadcast关系,数据类型需要与other满足数据类型推导规则,支持非连续的Tensor,
 * 数据格式支持ND；selfRef也是输出Tensor。
 * @param [in] other(aclTensor*): 数据类型支持UINT8,INT8,DOUBLE,FLOAT,INT32,INT64,INT16,FLOAT16,BOOL,BFLOAT16数据类型,
 * shape需要与other满足broadcast关系,数据类型需要与selfRef满足数据类型推导规则,支持非连续的Tensor,数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器,包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceGtTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGtTensor的第二段接口,用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小,由第一段接口aclnnGtTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器,包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceGtTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GTTENSOR_H_
