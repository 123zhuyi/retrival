/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef ACLNN_APPROXIMATE_EQUAL_H_
#define ACLNN_APPROXIMATE_EQUAL_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size for aclnnApproximateEqual (two-stage interface).
 * @param x1             [in]  Input tensor 1, dtype FLOAT, NDIM up to 8.
 * @param x2             [in]  Input tensor 2, same dtype/shape as x1.
 * @param tolerance      [in]  Non-negative finite float.
 * @param y              [out] Output tensor of dtype BOOL with shape == x1.
 * @param workspaceSize  [out] Required workspace in bytes (0 for this operator).
 * @param executor       [out] Created operator executor.
 */
ACLNN_API aclnnStatus aclnnApproximateEqualGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, float tolerance, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/** @brief Second-stage launch of ApproximateEqual. */
ACLNN_API aclnnStatus
aclnnApproximateEqual(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_APPROXIMATE_EQUAL_H_
