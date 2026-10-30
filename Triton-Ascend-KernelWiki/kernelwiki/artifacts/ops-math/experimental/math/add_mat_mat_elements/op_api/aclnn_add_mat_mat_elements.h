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
 * @file aclnn_add_mat_mat_elements.h
 * @brief ACLNN L2 API 接口声明 - AddMatMatElements 算子
 *
 * 计算公式：c_out = c × beta + alpha × a × b
 *
 * 两段式接口：
 *   1. aclnnAddMatMatElementsGetWorkspaceSize - 计算 workspace 大小，创建执行器
 *   2. aclnnAddMatMatElements - 执行计算
 */

#ifndef ACLNN_ADD_MAT_MAT_ELEMENTS_H_
#define ACLNN_ADD_MAT_MAT_ELEMENTS_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnAddMatMatElements 所需的 workspace 大小
 *
 * @param a          [in]  输入张量 a（与 b、c 同 shape 和 dtype）
 * @param b          [in]  输入张量 b
 * @param c          [in]  输入张量 c
 * @param alpha      [in]  标量缩放系数 alpha（aclScalar*，非空）
 * @param beta       [in]  标量缩放系数 beta（aclScalar*，非空）
 * @param cOut       [out]  输出张量（shape 与 a/b/c 相同）
 * @param workspaceSize [out] 返回所需 workspace 大小（字节）
 * @param executor   [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAddMatMatElementsGetWorkspaceSize(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclScalar* alpha, const aclScalar* beta,
    aclTensor* cOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 AddMatMatElements 算子计算
 *
 * @param workspace      [in] workspace 内存地址
 * @param workspaceSize  [in] workspace 大小（字节）
 * @param executor       [in] 执行器（由 GetWorkspaceSize 返回）
 * @param stream         [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus
aclnnAddMatMatElements(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ADD_MAT_MAT_ELEMENTS_H_
