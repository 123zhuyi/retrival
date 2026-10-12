/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_DIV_V3_H_
#define OP_API_INC_LEVEL2_ACLNN_DIV_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDivV3 第一段接口，计算 workspace 大小
 * @domain aclnn_math
 *
 * 计算公式：
 *   mode=0: $$ out = self / other $$
 *   mode=1: $$ out = trunc(self / other) $$
 *   mode=2: $$ out = floor(self / other) $$
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(self)] --> B([l0op::Contiguous])
 *   C[(other)] --> D([l0op::Contiguous])
 *   B --> E([l0op::BroadcastTo])
 *   D --> F([l0op::BroadcastTo])
 *   E --> G([l0op::DivV3])
 *   F --> G
 *   G --> H([l0op::ViewCopy])
 *   H --> I[(out)]
 * ```
 *
 * @param [in] self: 被除数张量
 * @param [in] other: 除数张量
 * @param [in] mode: 舍入模式 (0=RealDiv, 1=TruncDiv, 2=FloorDiv)
 * @param [in] out: 输出张量
 * @param [out] workspaceSize: 返回 workspace 大小
 * @param [out] executor: 返回 op 执行器
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnDivV3GetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, int64_t mode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnDivV3 第二段接口，执行计算
 */
ACLNN_API aclnnStatus aclnnDivV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_DIV_V3_H_
