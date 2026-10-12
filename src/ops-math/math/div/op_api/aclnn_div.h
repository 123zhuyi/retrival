/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DIV_H_
#define OP_API_INC_DIV_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDiv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与other一致。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要与self构成互相推导关系，shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，且数据格式需要与self一致。
 * @param [in]
 * mode：余数处理方式的整型常量，枚举值如下：0-默认不执行舍入。1-将除法的小数部分舍入为零。2-向下舍入除法的结果。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，且数据类型需要是self与other推导之后可转换的数据类型，shape需要是self与other
 * broadcast之后的shape，数据格式支持ND，且数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDivGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDivs的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnDivsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDivMod的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnDivModGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, int mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnDivMods的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnDivModsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, int mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceDiv的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceDivs的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceDivMod的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivModGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, int mode, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceDivMods的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, int mode, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDiv的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnDivGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus aclnnDivs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus aclnnDivMod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnDivMods(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnInplaceDiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnInplaceDivs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnInplaceDivMod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

ACLNN_API aclnnStatus
aclnnInplaceDivMods(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_DIV_H_
