/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_POLAR_H_
#define OP_API_INC_LEVEL2_ACLNN_POLAR_H_

#include "aclnn/aclnn_base.h"
// 注意：aclnn_util.h 含 ACLNN_API、CHECK_RET 等内部宏，但**只在 .cpp 内引用**；
// 这里不在公共头暴露 SDK 内部头，避免下游 build_aclnn.sh (CMakeLists 不含 SDK 内部 include path) 找不到。
// ACLNN_API 在 SDK 内是 visibility=default 宏，下游若未定义这里给个安全 fallback。
#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnPolar 第一段接口：参数校验 + Contiguous + BroadcastTo + 调 L0 kernel + ViewCopy。
 * @domain aclnn_math
 *
 * 公式： out = input * (cos(angle) + i*sin(angle))
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *   A[(input fp32)] --> B([l0op::Contiguous]) --> C([l0op::BroadcastTo])
 *   D[(angle fp32)] --> E([l0op::Contiguous]) --> F([l0op::BroadcastTo])
 *   C --> G([l0op::Polar])
 *   F --> G
 *   G --> H([l0op::ViewCopy]) --> I[(out complex64)]
 * ```
 *
 * @param [in]  input         模长分量，fp32
 * @param [in]  angle         幅角（弧度），fp32；与 input 满足 numpy 广播
 * @param [out] out           复数张量，complex64；shape 等于 input/angle 广播后的 shape
 * @param [out] workspaceSize device 侧需要的 workspace 字节数
 * @param [out] executor      op 执行器（第二段接口使用）
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor* input, const aclTensor* angle, aclTensor* out,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnPolar 第二段接口：执行计算
 */
ACLNN_API aclnnStatus aclnnPolar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_POLAR_H_
