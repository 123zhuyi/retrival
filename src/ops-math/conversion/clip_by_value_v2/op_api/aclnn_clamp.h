/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_CLAMP_H_
#define OP_API_INC_LEVEL2_ACLNN_CLAMP_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnClamp的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnClamp的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnClamp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnClampMin的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampMinGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnClampMin的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnClampMin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnClampTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnClampMinTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnClampMinTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* clipValueMin, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnClampMinTensor的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnClampMinTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnInplaceClampMinTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceClampMinTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* clipValueMin, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceClampMinTensor的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnInplaceClampMinTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnClampTensor的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus
aclnnClampTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

/**
 * @brief aclnnClampMax的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：将输入的所有元素限制在[-inf,max]范围内。
 * 计算公式：
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] self: 输入tensor，数据类型支持FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] clipValueMax: 上界，数据类型需要可转换成self的数据类型。
 * @param [in] out: 输出，数据类型支持FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64，
 * 且数据类型和self保持一致，shape和self保持一致，数据格式支持ND（[参考](#)）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnClampMaxGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMax, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnClampMax的第二段接口，用于执行计算。
 *
 * 算子功能：将输入的所有元素限制在[-inf,max]范围内。
 * 计算公式：
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnClampMaxGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnClampMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceClampMax的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：将输入的所有元素限制在[-inf,max]范围内。
 * 计算公式：
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] selfRef: 输入tensor，数据类型支持FLOAT16、FLOAT、FLOAT64、INT8、UINT8、INT16、INT32、INT64。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [in] clipValueMax: 上界，数据类型需要可转换成self的数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceClampMaxGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* clipValueMax, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceClampMax的第二段接口，用于执行计算。
 *
 * 算子功能：将输入的所有元素限制在[-inf,max]范围内。
 * 计算公式：
 * $$ {y}_{i} = min({{x}_{i}},max) $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceClampMaxGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceClampMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnClampMaxTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：将输入的所有元素限制在[-inf, max]范围内。
 *
 * @param [in] self: 输入tensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64，
 * 且数据类型需要与max的数据类型需满足数据类型推导规则，shape需要与max满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] max: 输入上限值tensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64
 * 且数据类型需要与self的数据类型需满足数据类型推导规则，shape需要与max满足broadcast关系。支持非连续的Tensor，数据格式支持ND。
 * @param [in] out: 输出tensor，数据类型支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64，
 * 且数据类型需要是self与max推导之后可转换的数据类型，shape需要是self与max
 * broadcast之后的shape。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnClampMaxTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* max, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceClampMaxTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceClampMaxTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* max, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnClampMaxTensor的第二段接口，用于执行计算。
 *
 * 算子功能：将输入的所有元素限制在[-inf, max]范围内。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnClampMaxTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnClampMaxTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnInplaceClampMaxTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_CLAMP_H_