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
 * @file aclnn_asinh_v2.h
 * @brief ACLNN L2 API 接口声明 - AsinhV2 算子
 *
 * ACLNN 接口采用两段式设计：
 * - aclnnAsinhV2GetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnnAsinhV2: 执行计算
 *
 * 支持 dtype（L2层）：
 *   float16, float32 -> 直接走 Kernel 路径
 *   int8, int16, int32, int64, uint8, bool, double -> op_api 层 Cast 到 float32 后走 Kernel 路径
 */

#ifndef ACLNN_ASINH_V2_H_
#define ACLNN_ASINH_V2_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnAsinhV2 所需的 workspace 大小
 * @param x [in] 输入张量（支持 float16/float32/int8/int16/int32/int64/uint8/bool/double）
 * @param out [in] 输出张量（float16 输入对应 float16，其余对应 float32）
 * @param workspaceSize [out] 返回所需 workspace 大小（通常为 0）
 * @param executor [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnAsinhV2GetWorkspaceSize(
    const aclTensor* x, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 AsinhV2 算子计算
 * @param workspace [in] workspace 内存地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 执行器
 * @param stream [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus
aclnnAsinhV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ASINH_V2_H_
