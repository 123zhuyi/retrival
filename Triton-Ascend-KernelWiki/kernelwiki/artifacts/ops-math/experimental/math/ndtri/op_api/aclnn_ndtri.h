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
 * @file aclnn_ndtri.h
 * @brief ACLNN L2 API 接口声明 - Ndtri（标准正态 CDF 反函数 / probit）
 *
 * 两段式接口：
 *   - aclnnNdtriGetWorkspaceSize: 参数检查、创建 executor、返回所需 workspace
 *   - aclnnNdtri: 执行计算
 */

#ifndef ACLNN_NDTRI_H_
#define ACLNN_NDTRI_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算 aclnnNdtri 所需的 workspace 大小。
 *
 * @param self          [in]  输入 Tensor (dtype ∈ {FLOAT, FLOAT16, BF16})；value range (0, 1)
 * @param out           [out] 输出 Tensor（shape 和 dtype 与 self 一致）
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor      [out] 算子执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus
aclnnNdtriGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 Ndtri 算子计算。
 */
ACLNN_API aclnnStatus aclnnNdtri(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_NDTRI_H_
