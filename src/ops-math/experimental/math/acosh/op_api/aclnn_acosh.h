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
 * @file aclnn_acosh.h
 * @brief ACLNN L2 API 接口声明 - Acosh 算子
 *
 * 两段式接口：
 * - aclnnAcoshGetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnnAcosh: 执行计算
 */

#ifndef ACLNN_ACOSH_H_
#define ACLNN_ACOSH_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnAcosh 所需的 workspace 大小
 * @param self [in]  输入张量 x（支持 fp16/fp32/bf16）
 * @param out  [in]  输出张量 y（dtype 与 self 相同，shape 与 self 相同）
 * @param workspaceSize [out] 返回所需 workspace 大小（字节）
 * @param executor [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAcoshGetWorkspaceSize(
    const aclTensor* self, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 Acosh 算子计算
 * @param workspace [in] workspace 内存地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 执行器
 * @param stream [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAcosh(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ACOSH_H_
