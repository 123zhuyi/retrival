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
 * @file aclnn_xlog1py.h
 * @brief ACLNN L2 API - xlog1py: z = x * log1p(y), with x==0 → z=0
 *
 * 两段式接口:
 * - aclnnXlog1pyGetWorkspaceSize: 计算 workspace
 * - aclnnXlog1py: 执行计算
 */

#ifndef ACLNN_XLOG1PY_H_
#define ACLNN_XLOG1PY_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnXlog1py的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：完成 x * log1p(y) 计算，x 为 0 时输出 0。
 * 参数描述：
 * @param [in] x: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BF16，shape需要与y满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，维度不大于8。
 * @param [in] y: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BF16，shape需要与x满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND，维度不大于8。
 * @param [in] z: npu
 * device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BF16，shape为x与y broadcast之后的shape。
 * 支持非连续的Tensor，数据格式支持ND，维度不大于8。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnXlog1pyGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *z,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief aclnnXlog1py的第二段接口，用于执行计算。
 *
 * 算子功能：完成 x * log1p(y) 计算，x 为 0 时输出 0。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnXlog1pyGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnXlog1py(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_XLOG1PY_H_
