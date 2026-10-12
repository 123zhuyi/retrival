/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FMOD_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_FMOD_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFmodTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：返回self除以other的余数。
 * 计算公式：$$ out_{i} = self_{i} - (other_{i} *\left \lfloor (self_{i}/other_{i}) \right \rfloor) $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(selfRef)] -->B([l0op::Contiguous])
 * B -->C([l0op::Cast])
 * C -->D([l0op::Mod])
 * E[(other)]-->F([l0op::Contiguous])
 * F -->H([l0op::Cast])
 * H --> D
 * D--> G([l0op::Cast])
 * G --> I([l0op::ViewCopy])
 * I --> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor
 * 数据类型支持DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UNIT8
 * 且数据类型与other的数据类型需满足数据类型推导规则。支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UNIT8
 * 且数据类型与self的数据类型需满足数据类型推导规则。支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: npu device侧的aclTensor，数据类型支持DOUBLE、BFLOAT16、FLOAT16、FLOAT32、INT32、INT64、INT8、UNIT8，
 * shape需要是self与other broadcast之后的shape。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFmodTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFmodTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnFmodTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnFmodTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceFmodTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：返回selfRef除以other的余数。
 * 计算公式：$$ out_{i} = self_{i} - (other_{i} *\left \lfloor (self_{i}/other_{i}) \right \rfloor) $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(selfRef)] -->B([l0op::Contiguous])
 * B -->C([l0op::Cast])
 * C -->D([l0op::Mod])
 * E[(other)]-->F([l0op::Contiguous])
 * F -->H([l0op::Cast])
 * H --> D
 * D--> G([l0op::Cast])
 * G --> I([l0op::ViewCopy])
 * I --> J[(out)]
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor
 * 数据类型支持DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UNIT8
 * 且数据类型与other的数据类型需满足数据类型推导规则。支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持DOUBLE、FLOAT16、FLOAT32、INT32、INT64、INT8、UNIT8
 * 且数据类型与self的数据类型需满足数据类型推导规则。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceFmodTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceFmodTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceFmodTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceFmodTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FMOD_TENSOR_H_
