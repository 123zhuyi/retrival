/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_LEFT_SHIFT_H_
#define OP_API_INC_LEVEL2_ACLNN_LEFT_SHIFT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLeftShift的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：输入张量self中每个元素，根据shiftBits对应位置的参数，按位进行左移。
 * 计算公式：
 * out_{i} = self_{i}<<shiftBits_{i}
 * @domain aclnn_math
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * 数据类型需要与shiftBits构成互相推导关系，self需要与shiftBits满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in]   shiftBits
 * 输入Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * 数据类型需要与self构成互相推导关系，shiftBits需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  out
 * 输出Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * shape需要与self和shiftBits做broadcast后的shape一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLeftShiftGetWorkspaceSize(
    const aclTensor* self, const aclTensor* shiftBits, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnLeftShift的第二段接口，用于执行计算。
 * 功能描述：输入张量self中每个元素，根据shiftBits对应位置的参数，按位进行左移。
 * 计算公式：
 * out_{i} = self_{i}<<shiftBits_{i}
 * @domain aclnn_math
 * 实现说明：
 * api计算的基本路径：
```mermaid
flowchart LR
   A[(self)]-->B([l0op::Contiguous])-->C([l0op::Cast])-->D[(l0op::BroadcastTo)]-->E([l0op::LeftShift])
   -->F([l0op::Cast])-->G([l0op::ViewCopy])-->H[(out)]
   I[(shiftBits)]-->J([l0op::Contiguous])-->K([l0op::Cast])-->L[(l0op::BroadcastTo)]-->E([l0op::LeftShift])
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLeftShiftGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLeftShift(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnLeftShifts的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：输入张量self中每个元素，根据输入标量shiftBits，按位进行左移。
 * 计算公式：
 * out_{i} = self_{i}<<shiftBits
 * @domain aclnn_math
 * 参数描述：
 * @param [in]   self
 * 输入Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * 数据类型需要与shiftBits构成互相推导关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in]   shiftBits
 * 输入Scalar，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * 数据类型需要与self构成互相推导关系。
 * @param [out]  out
 * 输出Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * shape需要与self和shiftBits做broadcast后的shape一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLeftShiftsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* shiftBits, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnLeftShifts的第二段接口，用于执行计算。
 * 功能描述：输入张量self中每个元素，根据输入标量shiftBits，按位进行左移。
 * 计算公式：
 * out_{i} = self_{i}<<shiftBits
 * @domain aclnn_math
 * 实现说明：
 * api计算的基本路径：
```mermaid
flowchart LR
   A[(self)]-->B([l0op::Contiguous])-->C([l0op::Cast])-->D([l0op::LeftShift])
   -->E([l0op::Cast])-->F([l0op::ViewCopy])-->G[(out)]
   H((shiftBits))-->I([l0op::Cast])-->J([l0op::BroadcastTo])-->D([l0op::LeftShift])
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLeftShiftsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLeftShifts(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_LEFT_SHIFT_H_
