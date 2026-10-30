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

/**
 * @file aclnn_log_space.h
 * @brief ACLNN L2 API - LogSpace
 */
#ifndef ACLNN_LOG_SPACE_H_
#define ACLNN_LOG_SPACE_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnLogSpace 所需的 workspace 大小
 * @param start [in] aclScalar，对数起始指数
 * @param end   [in] aclScalar，对数结束指数
 * @param steps [in] int64_t，输出元素数（>= 0）
 * @param base  [in] double，对数底（> 0）
 * @param result [in] 输出 Tensor，dtype 决定输出类型
 * @param workspaceSize [out]
 * @param executor [out]
 */
ACLNN_API aclnnStatus aclnnLogSpaceGetWorkspaceSize(
    const aclScalar* start, const aclScalar* end, int64_t steps, double base, const aclTensor* result,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnLogSpace(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
