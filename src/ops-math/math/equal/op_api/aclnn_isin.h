/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_IS_IN_H_
#define OP_API_INC_IS_IN_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIsInScalarTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：检查element是否在张量testElements中。
 *
 * @param [in] element: npu device侧的aclScalar，数据类型支持整型，浮点类型。
 * @param [in] testElements: host侧的aclTensor，数据类型支持整型，浮点类型。支持非连续的Tensor，数据格式支持ND
 * @param [in] assumeUnique: host侧的bool，表示testElements中的值是否唯一。
 * @param [in] invert: host侧的bool，表示输出结果是否反转。
 * @param [in] out: npu device侧的aclTensor，数据类型支持布尔。支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(
    const aclScalar* element, const aclTensor* testElements, bool assumeUnique, bool invert, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIsInScalarTensor的第二段接口，用于执行计算。
 *
 * 算子功能：检查element是否在张量testElements中。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnIsInScalarTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnIsInScalarTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_IS_IN_H_
