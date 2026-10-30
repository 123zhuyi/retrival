/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 */

#ifndef ACLNN_FRESNEL_SIN_H_
#define ACLNN_FRESNEL_SIN_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算 aclnnFresnelSin 算子所需的 workspace 大小并完成 executor 构建（两段式接口第一段）。
 *
 * 算子语义：逐元素计算 Fresnel 正弦积分 S(x) = ∫₀ˣ sin(π·t²/2) dt。
 *
 * @param [in]  x              输入张量，数据类型支持 FLOAT、FLOAT16、BFLOAT16；数据格式支持 ND；
 *                             shape 任意，不支持空 tensor 以外的非法 shape。
 * @param [out] out            输出张量，shape、dtype、format 与 x 一致。
 * @param [out] workspaceSize  返回需要在 Device 侧申请的 workspace 大小（字节）。
 * @param [out] executor       返回 op 执行器，供第二段接口使用。
 * @return aclnnStatus         执行状态码：成功返回 ACLNN_SUCCESS，否则返回相应错误码。
 */
ACLNN_API aclnnStatus
aclnnFresnelSinGetWorkspaceSize(const aclTensor* x, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 aclnnFresnelSin 算子计算（两段式接口第二段）。
 *
 * @param [in] workspace      Device 侧 workspace 起始地址，由调用者按 workspaceSize 申请。
 * @param [in] workspaceSize  workspace 大小（字节），需与第一段接口返回值一致。
 * @param [in] executor       op 执行器，由 aclnnFresnelSinGetWorkspaceSize 创建。
 * @param [in] stream         指定执行的 acl stream。
 * @return aclnnStatus        执行状态码：成功返回 ACLNN_SUCCESS，否则返回相应错误码。
 */
ACLNN_API aclnnStatus
aclnnFresnelSin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_FRESNEL_SIN_H_
