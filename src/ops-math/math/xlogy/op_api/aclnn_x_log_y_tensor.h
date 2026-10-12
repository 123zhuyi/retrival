/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_X_LOG_Y_TENSOR_H_
#define OP_API_INC_X_LOG_Y_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnXLogYTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnXLogYTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnXLogYTensor的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnXLogYTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * @brief aclnnInplaceXLogYTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 */
ACLNN_API aclnnStatus aclnnInplaceXLogYTensorGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other,
                                                              uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceXLogYTensor的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnInplaceXLogYTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                              aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_X_LOG_Y_TENSOR_H_