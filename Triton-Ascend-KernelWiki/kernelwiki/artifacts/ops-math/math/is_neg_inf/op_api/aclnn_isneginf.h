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
 * \file aclnn_isneginf.h
 * \brief
 */

#ifndef OP_API_INC_ISNEGINF_H_
#define OP_API_INC_ISNEGINF_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIsNegInf的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * 算子功能：判断输入张量的元素是否为负无穷。
 * 实现说明-计算图
 * 计算图：如下
 * 场景1：浮点数场景
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0op::Contiguous])
 *   B --> D([l0op::IsNegInf])
 *   D --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 * 场景2：非浮点数，有界，返回False
 * ```mermaid
 * graph LR
 *   A[(self)] -->B([l0op::Fill])
 *   B --> I([l0op::ViewCopy])
 *   I --> J[(out)]
 * ```
 *
 * @param [in] self：输入`self`,数据类型支持FLOAT、FLOAT16、BFLOAT16(Ascend910B),
 * INT32、INT64、INT16、INT8、UINT8、BOOL。支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [out] out：输出`out`,数据类型支持BOOL，shape需要与self一致。
 * 支持[非连续的Tensor](#)，数据格式支持ND（[参考](#)）。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnIsNegInfGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIsNegInf的第二段接口，用于执行计算
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnIsNegInfGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnIsNegInf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ISNEGINF_H_
