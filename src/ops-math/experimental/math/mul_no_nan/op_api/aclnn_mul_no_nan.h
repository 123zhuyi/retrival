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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_mul_no_nan.h
 * @brief ACLNN L2 API declaration for MulNoNan
 */

#ifndef ACLNN_MUL_NO_NAN_H_
#define ACLNN_MUL_NO_NAN_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute workspace size for aclnnMulNoNan
 * @param x [in] input tensor (multiplicand)
 * @param y [in] input tensor (multiplier, y=0 forces output to 0)
 * @param z [in] output tensor
 * @param workspaceSize [out] required workspace size
 * @param executor [out] op executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnMulNoNanGetWorkspaceSize(
    const aclTensor* x, const aclTensor* y, const aclTensor* z, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Execute MulNoNan computation
 * @param workspace [in] workspace memory
 * @param workspaceSize [in] workspace size
 * @param executor [in] op executor
 * @param stream [in] ACL stream
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus
aclnnMulNoNan(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_MUL_NO_NAN_H_
