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
 * @file aclnn_asin_with_agent.h
 * @brief ACLNN L2 API 接口声明 - AsinWithAgent 算子
 *
 * 两段式 ACLNN 接口：
 *   - aclnnAsinWithAgentGetWorkspaceSize: 计算 workspace 大小，创建执行器
 *   - aclnnAsinWithAgent:                执行计算
 */

#ifndef ACLNN_ASIN_WITH_AGENT_H_
#define ACLNN_ASIN_WITH_AGENT_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnAsinWithAgent 所需的 workspace 大小
 *
 * @param x            [in]  输入张量，dtype: FLOAT/FLOAT16/DOUBLE/INT8/INT16/INT32/INT64/UINT8/BOOL
 * @param y            [in]  输出张量，dtype: 与 x 相同（浮点）或 FLOAT32（整数/BOOL）
 * @param workspaceSize [out] 返回所需 workspace 大小（字节）
 * @param executor     [out] 返回算子执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus
aclnnAsinWithAgentGetWorkspaceSize(const aclTensor* x, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 AsinWithAgent 算子计算
 *
 * @param workspace     [in] workspace 内存地址（由框架管理）
 * @param workspaceSize [in] workspace 大小
 * @param executor      [in] 执行器（由 GetWorkspaceSize 创建）
 * @param stream        [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus
aclnnAsinWithAgent(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ASIN_WITH_AGENT_H_
