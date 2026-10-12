/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_logspace.h
 * \brief
 */
#ifndef OP_API_INC_LOGSPACED_H_
#define OP_API_INC_LOGSPACED_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLogSpace的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnnop_ops_train
 * 算子功能： 返回在一个区间[base^start, base^end]上均匀分布的对数间隔的一维张量
 * @param [in] start: 起始值的标量指针
 * @param [in] end: 结束值的标量指针
 * @param [in] steps: 输出张量中的元素数量
 * @param [in] base: 对数的底数
 * @param [in] result: 输出张量，数据类型支持FLOAT32,FLOAT16,BFLOAT16
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnLogSpaceGetWorkspaceSize(
    const aclScalar *start, const aclScalar *end, int64_t steps, double base,
    const aclTensor *result, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief: aclnnLogSpace的第二段接口，用于执行计算
 * @domain aclnnop_ops_train
 * 算子功能： 返回在一个区间[base^start, base^end]上均匀分布的对数间隔的一维张量
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLogSpaceDGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLogSpace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LOGSPACE_H_