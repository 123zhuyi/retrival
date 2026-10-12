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
 * @file aclnn_atan_grad.h
 * @brief ACLNN L2 API 接口声明 - AtanGrad 算子
 *
 * AtanGrad 计算：dx = dy * (1 / (1 + x^2))
 *
 * 两段式接口：
 *   1. aclnnAtanGradGetWorkspaceSize - 计算 workspace 大小，创建执行器
 *   2. aclnnAtanGrad                 - 执行计算
 */

#ifndef ACLNN_ATAN_GRAD_H_
#define ACLNN_ATAN_GRAD_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnAtanGrad 所需的 workspace 大小
 * @param x            [in]  正向输入张量
 * @param dy           [in]  上游梯度张量
 * @param dx           [in]  输出梯度张量（由调用方分配）
 * @param workspaceSize [out] 返回所需 workspace 大小（字节）
 * @param executor     [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAtanGradGetWorkspaceSize(
    const aclTensor* x, const aclTensor* dy, const aclTensor* dx, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 AtanGrad 算子计算
 * @param workspace     [in] workspace 内存地址（可为 nullptr 当 workspaceSize=0）
 * @param workspaceSize [in] workspace 大小
 * @param executor      [in] 执行器
 * @param stream        [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus
aclnnAtanGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ATAN_GRAD_H_
